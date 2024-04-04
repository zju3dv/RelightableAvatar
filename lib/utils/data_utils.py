import os
import cv2
import h5py
import torch
import imageio
import asyncio
import numpy as np

from tqdm import tqdm
from PIL import Image
from functools import lru_cache

# from imgaug import augmenters as iaa
from typing import Tuple, Union, List

from torch.nn import functional as F
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils.collate import default_collate, default_convert

from lib.utils.net_utils import get_rigid_transform_nobatch as net_get_rigid_transform
from lib.utils.base_utils import dotdict
from lib.utils.log_utils import log


def as_torch_func(func):
    def wrapper(*args, **kwargs):
        args = to_numpy(args)
        kwargs = to_numpy(kwargs)
        ret = func(*args, **kwargs)
        return to_tensor(ret)
    return wrapper

def as_numpy_func(func):
    def wrapper(*args, **kwargs):
        args = to_tensor(args)
        kwargs = to_tensor(kwargs)
        ret = func(*args, **kwargs)
        return to_numpy(ret)
    return wrapper


def variance_of_laplacian(image: np.ndarray):
    if image.ndim == 3 and image.shape[-1] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def alpha2sdf(alpha, beta, dists=0.005):
    return beta * np.log(2 * beta * (-np.log(1 - alpha) / dists))


def h5_to_dotdict(h5: h5py.File) -> dotdict:
    d = {key: h5_to_dotdict(h5[key]) if isinstance(h5[key], h5py.Group) else h5[key][:] for key in h5.keys()}  # loaded as numpy array
    d = dotdict(d)
    return d


def h5_to_list_of_dotdict(h5: h5py.File) -> list:
    return [h5_to_dotdict(h5[key]) for key in tqdm(h5)]


def to_h5py(value, h5: h5py.File, key: str = None, compression: str = 'gzip'):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        h5.create_dataset(str(key), data=value, compression=compression)
    elif isinstance(value, list):
        if key is not None:
            h5 = h5.create_group(str(key))
        [to_h5py(v, h5, k) for k, v in enumerate(value)]
    elif isinstance(value, dict):
        if key is not None:
            h5 = h5.create_group(str(key))
        [to_h5py(v, h5, k) for k, v in value.items()]
    else:
        raise NotImplementedError(f'unsupported type to write to h5: {type(value)}')


def export_h5(batch: dotdict, filename):
    with h5py.File(filename, 'w') as f:
        to_h5py(batch, f)


def load_h5(filename):
    with h5py.File(filename, 'r') as f:
        return h5_to_dotdict(f)


def merge_faces(faces, *args):
    # Copied from trimesh, this will select one uv coordinates for a particular vertex
    """
    Textured meshes can come with faces referencing vertex
    indices (`v`) and an array the same shape which references
    vertex texture indices (`vt`) and sometimes even normal (`vn`).

    Vertex locations with different values of any of these can't
    be considered the "same" vertex, and for our simple data
    model we need to not combine these vertices.

    Parameters
    -------------
    faces : (n, d) int
      References vertex indices
    *args : (n, d) int
      Various references of corresponding values
      This is usually UV coordinates or normal indexes
    maintain_faces : bool
      Do not alter original faces and return no-op masks.

    Returns
    -------------
    new_faces : (m, d) int
      New faces for masked vertices
    mask_v : (p,) int
      A mask to apply to vertices
    mask_* : (p,) int
      A mask to apply to vt array to get matching UV coordinates
      Returns as many of these as args were passed
    """

    # start with not altering faces at all
    result = [faces]
    # find the maximum index referenced by faces
    max_idx = faces.max()
    # add a vertex mask which is just ordered
    result.append(np.arange(max_idx + 1))

    # now given the order is fixed do our best on the rest of the order
    for arg in args:
        # create a mask of the attribute-vertex mapping
        # note that these might conflict since we're not unmerging
        masks = np.zeros((3, max_idx + 1), dtype=np.int64)
        # set the mask using the unmodified face indexes
        for i, f, a in zip(range(3), faces.T, arg.T):
            masks[i][f] = a
        # find the most commonly occurring attribute (i.e. UV coordinate)
        # and use that index note that this is doing a float conversion
        # and then median before converting back to int: could also do this as
        # a column diff and sort but this seemed easier and is fast enough
        result.append(np.median(masks, axis=0).astype(np.int64))

    return result


def get_mesh(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, colors: torch.Tensor = None, normals: torch.Tensor = None, filename: str = "default.ply"):
    from trimesh import Trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial, SimpleMaterial
    from lib.utils.mesh_utils import face_normals, loop_subdivision

    verts, faces = to_numpy([verts, faces])
    verts = verts.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    # MARK: used process=False here to preserve vertex order
    mesh = Trimesh(verts, faces, process=False)
    if colors is None:
        # colors = verts
        colors = face_normals(torch.from_numpy(verts), torch.from_numpy(faces).long()) * 0.5 + 0.5
    colors = to_numpy(colors)
    colors = colors.reshape(-1, 3)
    colors = (np.concatenate([colors, np.ones([*colors.shape[:-1], 1])], axis=-1) * 255).astype(np.uint8)
    if len(verts) == len(colors):
        mesh.visual.vertex_colors = colors
    elif len(faces) == len(colors):
        mesh.visual.face_colors = colors

    if normals is not None:
        normals = to_numpy(normals)
        mesh.vertex_normals = normals

    if uv is not None:
        uv = to_numpy(uv)
        uv = uv.reshape(-1, 2)
        img = to_numpy(img)
        img = img.reshape(*img.shape[-3:])
        img = Image.fromarray(np.uint8(img * 255))
        mat = SimpleMaterial(
            image=img,
            diffuse=(0.8, 0.8, 0.8),
            ambient=(1.0, 1.0, 1.0),
        )
        mat.name = os.path.splitext(os.path.split(filename)[1])[0]
        texture = TextureVisuals(uv=uv, material=mat)
        mesh.visual = texture

    return mesh


def get_tensor_mesh_data(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, uvfaces: torch.Tensor = None):

    # pytorch3d wants a tensor
    verts, faces, uv, img, uvfaces = to_tensor([verts, faces, uv, img, uvfaces])
    verts = verts.view(-1, 3)
    faces = faces.view(-1, 3)
    uv = uv.view(-1, 2)
    img = img.view(img.shape[-3:])
    uvfaces = uvfaces.view(-1, 3)

    # textures = TexturesUV(img, uvfaces, uv)
    # meshes = Meshes(verts, faces, textures)
    return verts, faces, uv, img, uvfaces


def export_dotdict(batch: dotdict, filename):
    batch = to_numpy(batch)
    np.savez_compressed(filename, **batch)


def load_mesh(filename: str, device='cuda', load_uv=False, load_aux=False):
    from pytorch3d.io import load_ply, load_obj
    vm, fm = None, None
    if filename.endswith('.npz'):
        mesh = np.load(filename)
        v = torch.from_numpy(mesh['verts'])
        f = torch.from_numpy(mesh['faces'])

        if load_uv:
            vm = torch.from_numpy(mesh['uvs'])
            fm = torch.from_numpy(mesh['uvfaces'])
    else:
        if filename.endswith('.ply'):
            v, f = load_ply(filename)
        elif filename.endswith('.obj'):
            v, faces_attr, aux = load_obj(filename)
            f = faces_attr.verts_idx

            if load_uv:
                vm = aux.verts_uvs
                fm = faces_attr.textures_idx
        else:
            raise NotImplementedError(f'Unrecognized input format for: {filename}')

    v = v.to(device, non_blocking=True).contiguous()
    f = f.to(device, non_blocking=True).contiguous()

    if load_uv:
        vm = vm.to(device, non_blocking=True).contiguous()
        fm = fm.to(device, non_blocking=True).contiguous()

    if load_uv:
        if load_aux:
            return v, f, vm, fm, aux
        else:
            return v, f, vm, fm
    else:
        return v, f


def export_mesh(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, uvfaces: torch.Tensor = None, colors: torch.Tensor = None, normals: torch.Tensor = None, filename: str = "default.ply", subdivision=0):
    if subdivision > 0:
        from lib.utils.mesh_utils import face_normals, loop_subdivision
        verts, faces = loop_subdivision(verts, faces, subdivision)

    if filename.endswith('.npz'):
        def collect_args(**kwargs): return kwargs
        kwargs = collect_args(verts=verts, faces=faces, uv=uv, img=img, uvfaces=uvfaces, colors=colors, normals=normals)
        ret = dotdict({k: v for k, v in kwargs.items() if v is not None})
        export_dotdict(ret, filename)

    elif filename.endswith('.ply') or filename.endswith('.obj'):
        if uvfaces is None:
            mesh = get_mesh(verts, faces, uv, img, colors, normals, filename)
            mesh.export(filename)
        else:
            from pytorch3d.io import save_obj
            verts, faces, uv, img, uvfaces = get_tensor_mesh_data(verts, faces, uv, img, uvfaces)
            save_obj(filename, verts, faces, verts_uvs=uv, faces_uvs=uvfaces, texture_map=img)
    else:
        raise NotImplementedError(f'Unrecognized input format for: {filename}')


def export_pynt_pts_alone(pts, rgb=None, filename="default.ply"):
    import pandas as pd
    from pyntcloud import PyntCloud
    data = {}

    pts = pts if isinstance(pts, np.ndarray) else pts.detach().cpu().numpy()
    pts = pts.reshape(-1, 3)
    data['x'] = pts[:, 0].astype(np.float32)
    data['y'] = pts[:, 1].astype(np.float32)
    data['z'] = pts[:, 2].astype(np.float32)

    if rgb is not None:
        rgb = rgb if isinstance(rgb, np.ndarray) else rgb.detach().cpu().numpy()
        rgb = rgb.reshape(-1, 3)
        data['red'] = rgb[:, 0].astype(np.uint8)
        data['green'] = rgb[:, 1].astype(np.uint8)
        data['blue'] = rgb[:, 2].astype(np.uint8)
    else:
        data['red'] = (pts[:, 0] * 255).astype(np.uint8)
        data['green'] = (pts[:, 1] * 255).astype(np.uint8)
        data['blue'] = (pts[:, 2] * 255).astype(np.uint8)

    df = pd.DataFrame(data)

    cloud = PyntCloud(df)  # construct the data
    return cloud.to_file(filename)


def export_pynt_pts(pts: torch.Tensor, rgb: torch.Tensor = None, normals: torch.Tensor = None, filename: str = "default.ply"):
    from pandas import DataFrame
    from pyntcloud import PyntCloud

    def to_numpy(batch) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            batch = [to_numpy(b) for b in batch]
        elif isinstance(batch, dict):
            batch = dotdict({k: (to_numpy(v) if k != "meta" else v) for k, v in batch.items()})
        elif isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().numpy()
        else:  # numpy and others
            batch = np.array(batch)
        return batch

    data = dotdict()

    pts = to_numpy(pts)
    pts = pts.reshape(-1, 3)
    data.x = pts[:, 0].astype(np.float32)
    data.y = pts[:, 1].astype(np.float32)
    data.z = pts[:, 2].astype(np.float32)

    if rgb is not None:
        rgb = to_numpy(rgb)
        rgb = rgb.reshape(-1, 3)
        data.red = (rgb[:, 0] * 255).astype(np.uint8)
        data.green = (rgb[:, 1] * 255).astype(np.uint8)
        data.blue = (rgb[:, 2] * 255).astype(np.uint8)
    else:
        data.red = (pts[:, 0] * 255).astype(np.uint8)
        data.green = (pts[:, 1] * 255).astype(np.uint8)
        data.blue = (pts[:, 2] * 255).astype(np.uint8)

    if normals is not None:
        normals = to_numpy(normals)
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-13)
        normals = normals.reshape(-1, 3)
        data.nx = normals[:, 0].astype(np.float32)
        data.ny = normals[:, 1].astype(np.float32)
        data.nz = normals[:, 2].astype(np.float32)

    df = DataFrame(data)

    cloud = PyntCloud(df)  # construct the data
    return cloud.to_file(filename)


def export_o3d_pts(pts: torch.Tensor, filename: str = "default.ply"):
    import open3d as o3d
    pts = to_numpy(pts)
    pts = pts.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return o3d.io.write_point_cloud(filename, pcd)


def export_o3d_pcd(pts: torch.Tensor, rgb: torch.Tensor, normal: torch.Tensor, filename="default.ply"):
    import open3d as o3d
    pts, rgb, normal = to_numpy([pts, rgb, normal])
    pts = pts.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    normal = normal.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    return o3d.io.write_point_cloud(filename, pcd)


def export_pynt_pcd(pts: torch.Tensor, rgb: torch.Tensor, occ: torch.Tensor, filename="default.ply"):
    import pandas as pd
    from pyntcloud import PyntCloud
    pts, rgb, occ = to_numpy([pts, rgb, occ])
    pts = pts.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    occ = occ.reshape(-1, 1)
    # FIXME: cloud compare bad, set first to 0, last to 1
    for i in range(3):
        rgb[0, i] = 0
        rgb[-1, i] = 1
    occ[0, 0] = 0
    occ[-1, 0] = 1

    data = dotdict()
    data.x = pts[:, 0]
    data.y = pts[:, 1]
    data.z = pts[:, 2]
    # TODO: maybe, for compability, save color as uint?
    # currently saving as float number from [0, 1]
    data.red = rgb[:, 0]
    data.green = rgb[:, 1]
    data.blue = rgb[:, 2]
    data.alpha = occ[:, 0]

    # FIXME: we're saving extra data for loading in CloudCompare
    # can't assign same property to multiple fields
    data.r = rgb[:, 0]
    data.g = rgb[:, 1]
    data.b = rgb[:, 2]
    data.a = occ[:, 0]
    df = pd.DataFrame(data)

    cloud = PyntCloud(df)  # construct the data
    return cloud.to_file(filename)


export_pts = export_pynt_pts
export_pcd = export_pynt_pcd


def load_rgb_image(img_path) -> np.ndarray:
    # return cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1].copy()  # removing the stride (for conversion to tensor)
    return cv2.imread(img_path, cv2.IMREAD_COLOR)[..., [2, 1, 0]]  # BGR to RGB


def load_unchanged_image(img_path) -> np.ndarray:
    return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


def load_npz(index, folder):
    path = os.path.join(folder, f"{index}.npz")
    data = np.load(path)
    return dotdict({**data})


def load_dotdict(path):
    f = np.load(path)
    f = dotdict({**f})
    return f


def start_save_npz(index, dir, param: dict, remove_batch=True):
    return asyncio.create_task(async_save_npz(index, dir, param, remove_batch))


async def async_save_npz(index, dir, param: dict, remove_batch=True):
    log(f"Trying to save: {index}")
    save_npz(index, dir, param, remove_batch)


def save_img(index, dir, img: torch.Tensor, remove_batch=True, remap=False, flip=False):

    img = to_numpy(img)

    if remap:
        img *= 255
        img = img.astype(np.uint8)
    if flip:
        img = img[..., ::-1]

    if remove_batch:
        n_batch = img.shape[0]
        for b in range(n_batch):
            file_path = os.path.join(dir, f"{index*n_batch + b}.png")
            im = img[b]
            cv2.imwrite(file_path, im)
    else:
        file_path = os.path.join(dir, f"{index}.png")
        cv2.imwrite(file_path, img)


def save_npz(index, dir, param: dict, remove_batch=False):
    param = to_numpy(param)
    if remove_batch:
        n_batch = param[next(iter(param))].shape[0]
        for b in range(n_batch):
            file_path = os.path.join(dir, f"{index*n_batch + b}.npz")
            p = {k: v[b] for k, v in param.items()}
            np.savez_compressed(file_path, **p)
    else:
        file_path = os.path.join(dir, f"{index}.npz")
        np.savez_compressed(file_path, **param)


def to_cuda(batch, device="cuda") -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = [to_cuda(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_cuda(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device, non_blocking=True)
    else:  # numpy and others
        batch = torch.tensor(batch, device=device)
    return batch


def to_tensor(batch) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)):
        batch = [to_tensor(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_tensor(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        pass
    else:  # numpy and others
        batch = torch.tensor(batch)
    return batch


def to_cpu(batch, non_blocking=True) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = [to_cpu(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_cpu(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking)
    else:  # numpy and others
        batch = torch.tensor(batch, device="cpu")
    return batch


def to_numpy(batch, non_blocking=True) -> np.ndarray:
    if isinstance(batch, (tuple, list)):
        batch = [to_numpy(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_numpy(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.array(batch)
    return batch


def remove_batch(batch) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        batch = [remove_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (remove_batch(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, (torch.Tensor, np.ndarray)):  # numpy and others
        batch = batch[0]
    else:
        batch = np.array(batch)[0]
    return batch


def add_batch(batch) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        batch = [add_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (add_batch(v) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, (torch.Tensor, np.ndarray)):  # numpy and others
        batch = batch[None]
    else:
        batch = np.array(batch)[None]
    return batch


def add_iter_step(batch, iter_step) -> Union[torch.Tensor, np.ndarray]:
    return add_scalar(batch, iter_step, name="iter_step")


def add_scalar(batch, value, name) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        for b in batch:
            add_scalar(b, value, name)

    if isinstance(batch, dict):
        batch[name] = torch.tensor(value)
        batch['meta'][name] = torch.tensor(value)
    return batch


def get_voxel_grid_and_update_bounds(voxel_size: Union[List, np.ndarray], bounds: Union[List, np.ndarray]):
    # now here's the problem
    # 1. if you want the voxel size to be accurate, you bounds need to be changed along with this sampling process
    #    since the F.grid_sample will treat the bounds based on align_corners=True or not
    #    say we align corners, the actual bound on the sampled tpose blend weight should be determined by the actual sampling voxels
    #    not the bound that we kind of used to produce the voxels, THEY DO NOT LINE UP UNLESS your bounds is divisible by the voxel size in every direction
    # TODO: is it possible to somehow get rid of this book-keeping step
    if isinstance(voxel_size, List):
        voxel_size = np.array(voxel_size)
        bounds = np.array(bounds)
    # voxel_size: [0.005, 0.005, 0.005]
    # bounds: n_batch, 2, 3, initial bounds
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0] / 2, voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1] / 2, voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2] / 2, voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).astype(np.float32)
    bounds = np.stack([pts[0, 0, 0], pts[-1, -1, -1]], axis=0).astype(np.float32)
    return pts, bounds


def get_rigid_transform(pose: np.ndarray, joints: np.ndarray, parents: np.ndarray):
    # pose: N, 3
    # joints: N, 3
    # parents: N
    pose, joints, parents = default_convert([pose, joints, parents])
    J, A = net_get_rigid_transform(pose, joints, parents)
    J, A = to_numpy([J, A])

    return J, A


def logits_to_prob(logits):
    ''' Returns probabilities for logits
    Args:
        logits (tensor): logits
    '''
    odds = np.exp(logits)
    probs = odds / (1 + odds)
    return probs


def prob_to_logits(probs, eps=1e-4):
    ''' Returns logits for probabilities.
    Args:
        probs (tensor): probability tensor
        eps (float): epsilon value for numerical stability
    '''
    probs = np.clip(probs, a_min=eps, a_max=1 - eps)
    logits = np.log(probs / (1 - probs))
    return logits


def get_bounds(xyz, padding=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= padding
    max_xyz += padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    return bounds


def load_image(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('RGB', (int(im.width * ratio), int(im.height * ratio)))
        return np.asarray(im).astype(np.float32) / 255
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.ndim >= 3 and image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        elif image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32) / 255  # BGR to RGB
        height, width = image.shape[:2]
        if ratio != 1.0:
            image = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
        return image


def load_unchanged(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('RGB', (int(im.width * ratio), int(im.height * ratio)))
        return np.asarray(im)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        height, width = image.shape[:2]
        if ratio != 1.0:
            image = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_NEAREST)
        return image


def load_mask(msk_path: str, ratio=1.0):
    if msk_path.endswith('.jpg'):
        msk = Image.open(msk_path)
        msk.draft('L', (int(msk.width * ratio), int(msk.height * ratio)))
        msk = np.asarray(msk).astype(int)  # read the actual file content from drafted disk
        msk = msk * 255 / msk.max()  # if max already 255, do nothing
        msk = msk[..., None] > 128
        msk = msk.astype(np.uint8)
        return msk
    else:
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE).astype(int)  # BGR to GRAY
        msk = msk * 255 / msk.max()  # if max already 255, do nothing
        msk = msk[..., None] > 128  # make it binary
        msk = msk.astype(np.uint8)
        height, width = msk.shape[:2]
        if ratio != 1.0:
            msk = cv2.resize(msk.astype(np.uint8), (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_NEAREST)[..., None]
            # WTF: https://stackoverflow.com/questions/68502581/image-channel-missing-after-resizing-image-with-opencv
        return msk


def save_unchanged(img_path: str, img: np.ndarray, quality=100):
    if img.shape[-1] >= 3:
        img[..., :3] = img[..., [2, 1, 0]]
    if img_path.endswith('.hdr'):
        return cv2.imwrite(img_path, img)  # nothing to say about hdr
    if os.path.dirname(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_image(img_path: str, img: np.ndarray, jpeg_quality=100, png_dtype=np.uint16):
    if img.shape[-1] >= 3:
        img[..., :3] = img[..., [2, 1, 0]]
    if os.path.dirname(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    if img_path.endswith('.png'):
        max = np.iinfo(png_dtype).max
        img = (img * max).clip(0, max).astype(png_dtype)
    elif img_path.endswith('.jpg'):
        img = img[..., :3]  # only color
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img_path.endswith('.hdr'):
        img = img[..., :3]  # only color
    elif img_path.endswith('.exr'):
        # ... https://github.com/opencv/opencv/issues/21326
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    else:
        # should we try to discard alpha channel here?
        # exr could store alpha channel
        pass  # no transformation for other unspecified file formats
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def save_mask(msk_path: str, msk: np.ndarray, quality=100):
    if os.path.dirname(msk_path):
        os.makedirs(os.path.dirname(msk_path), exist_ok=True)
    if msk.ndim == 2:
        msk = msk[..., None]
    return cv2.imwrite(msk_path, msk[..., 0] * 255, [cv2.IMWRITE_JPEG_QUALITY, quality])


def list_to_numpy(x: list): return np.stack(x).transpose(0, 3, 1, 2)


def numpy_to_list(x: np.ndarray): return [y for y in x.transpose(0, 2, 3, 1)]


def list_to_tensor(x: list, device='cuda'): return torch.from_numpy(list_to_numpy(x)).to(device, non_blocking=True)  # convert list of numpy arrays of HWC to BCHW


def tensor_to_list(x: torch.Tensor): return numpy_to_list(x.detach().cpu().numpy())  # convert tensor of BCHW to list of numpy arrays of HWC


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def read_mask_by_img_path(data_root: str, img_path: str, erode_dilate_edge: bool = False, mask: str = '') -> np.ndarray:
    def read_mask_file(path):
        msk = load_mask(path).astype(np.uint8)
        if len(msk.shape) == 3:
            msk = msk[..., 0]
        return msk

    if mask:
        msk_path = os.path.join(data_root, img_path.replace('images', mask))
        if not os.path.exists(msk_path):
            msk_path = os.path.join(data_root, img_path.replace('images', mask)) + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(data_root, img_path.replace('images', mask))[:-4] + '.png'
        if not os.path.exists(msk_path):
            log(f'warning: defined mask path {msk_path} does not exist', 'yellow')
    else:
        msk_path = os.path.join(data_root, 'mask', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, 'mask', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, 'mask_cihp', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'merged_mask'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'rvm'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'rvm'))[:-4] + '.jpg'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'mask'))[:-4] + '.png'
    if not os.path.exists(msk_path):  # background matte v2
        msk_path = os.path.join(data_root, img_path.replace('images', 'bgmt'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'mask'))[:-4] + '.jpg'
    if not os.path.exists(msk_path):
        log(f'cannot find mask file: {msk_path}, using all ones', 'yellow')
        img = load_unchanged_image(os.path.join(data_root, img_path))
        msk = np.ones_like(img[:, :, 0]).astype(np.uint8)
        return msk

    msk = read_mask_file(msk_path)
    # erode edge inconsistence when evaluating and training
    if erode_dilate_edge:  # eroding edge on matte might erode the actual human
        msk = fill_mask_edge_with(msk)

    return msk


def fill_mask_edge_with(msk, border=5, value=100):
    msk = msk.copy()
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = value
    return msk


def get_rays_within_bounds_rendering(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    near = near.reshape(H, W)
    far = far.reshape(H, W)
    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)
    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays(H, W, K, R, T, subpixel=False):
    # calculate the camera origin
    ray_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(H, dtype=np.float32),
                       np.arange(W, dtype=np.float32),
                       indexing='ij')
    # 0->H, 0->W
    xy1 = np.stack([j, i, np.ones_like(i)], axis=2)
    if subpixel:
        rand = np.random.rand(H, W, 2) - 0.5
        xy1[:, :, :2] += rand
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    ray_d = pixel_world - ray_o[None, None]
    ray_d = ray_d / np.linalg.norm(ray_d, axis=2, keepdims=True)
    ray_o = np.broadcast_to(ray_o, ray_d.shape)
    return ray_o, ray_d


def get_near_far(bounds, ray_o, ray_d):
    """
    calculate intersections with 3d bounding box
    return: near, far (indexed by mask_at_box (bounding box mask))
    """
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box


def get_full_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near / norm_d[..., 0]
    far = far / norm_d[..., 0]
    return near, far, mask_at_box


def full_sample_ray(img, msk, K, R, T, bounds, split='train', subpixel=False):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T, subpixel)
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    msk = msk * mask_at_box
    coord = np.argwhere(np.ones_like(mask_at_box))  # every pixel
    ray_o = ray_o[coord[:, 0], coord[:, 1]].astype(np.float32)
    ray_d = ray_d[coord[:, 0], coord[:, 1]].astype(np.float32)
    near = near[coord[:, 0], coord[:, 1]].astype(np.float32)
    far = far[coord[:, 0], coord[:, 1]].astype(np.float32)
    rgb = img[coord[:, 0], coord[:, 1]].astype(np.float32)
    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_ray(img, msk, K, R, T, bounds, nrays, split='train', subpixel=False, body_ratio=0.5, face_ratio=0.0):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T, subpixel)
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    msk = msk * mask_at_box
    # if not len(np.argwhere(msk == 1)):
    #     from lib.utils.log_utils import stop_live_table
    #     stop_live_table()
    #     __import__('ipdb').set_trace()
    if "train" in split:
        n_body = int(nrays * body_ratio)
        n_face = int(nrays * face_ratio)
        n_rays = nrays - n_body - n_face
        coord_body = np.argwhere(msk == 1)
        coord_face = np.argwhere(msk == 13)
        coord_rand = np.argwhere(mask_at_box == 1)
        # if not len(coord_body): breakpoint()
        coord_body = coord_body[np.random.randint(len(coord_body), size=[n_body, ])]
        coord_face = coord_face[np.random.randint(len(coord_face), size=[n_face, ])]
        coord_rand = coord_rand[np.random.randint(len(coord_rand), size=[n_rays, ])]
        coord = np.concatenate([coord_body, coord_face, coord_rand], axis=0)
        mask_at_box = mask_at_box[coord[:, 0], coord[:, 1]]  # always True when training
    else:
        coord = np.argwhere(mask_at_box == 1)
        # will not modify mask at box
    ray_o = ray_o[coord[:, 0], coord[:, 1]].astype(np.float32)
    ray_d = ray_d[coord[:, 0], coord[:, 1]].astype(np.float32)
    near = near[coord[:, 0], coord[:, 1]].astype(np.float32)
    far = far[coord[:, 0], coord[:, 1]].astype(np.float32)
    rgb = img[coord[:, 0], coord[:, 1]].astype(np.float32)
    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_rays_within_bounds(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays_with_patch(H, W, K, R, T, bounds, patchsize):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    mask_at_box = mask_at_box.reshape(H, W)

    box_mask_inds = np.argwhere(mask_at_box)  # 2D -> 2D indexing, list of 2, (N, 2)
    box_min, box_max = (np.array([box_mask_inds[:, 0].min(),
                                 box_mask_inds[:, 1].min()]),
                        np.array([box_mask_inds[:, 0].max(),
                                  box_mask_inds[:, 1].max()]))
    # patchsize is the maximum
    patchsize = min(patchsize, *(box_max - box_min))

    i = np.random.randint(box_min[0], box_max[0] - patchsize)
    j = np.random.randint(box_min[1], box_max[1] - patchsize)
    mask = np.zeros_like(mask_at_box, dtype=bool)
    mask[i:i + patchsize, j:j + patchsize] = True

    full_mask = mask & mask_at_box  # get full mask, including mask_at_box and mask, still (512 * 512)
    box_mask = mask[mask_at_box]  # get valid indices in mask_at_box (mask_size,)
    mask_at_box = mask_at_box[mask]  # get valid mask_at_box value in (128, 128)
    near = near[box_mask].astype(np.float32)
    far = far[box_mask].astype(np.float32)
    ray_o = ray_o[full_mask].astype(np.float32)
    ray_d = ray_d[full_mask].astype(np.float32)

    return ray_o, ray_d, near, far, full_mask, mask_at_box


def get_rays_with_downscaling(H, W, K, R, T, bounds, split, downscale: int = 16, samples: int = 2):
    ray_o, ray_d = get_rays(H, W, K, R, T)
    if split != 'train':
        downscale = 1
        samples = 1

    h, w = H // downscale, W // downscale
    assert h * downscale == H and w * downscale == W, f"H, W: {H}, {W}, {downscale}, {samples}"
    assert samples <= downscale**2, f"H, W: {H}, {W}, {downscale}, {samples}"

    mask = np.zeros([H, W], dtype=bool)
    for i in range(h):
        for j in range(w):
            choices = np.random.choice(downscale**2, size=samples, replace=False)
            for c in choices:
                l = c // downscale
                k = c % downscale
                mask[i * downscale + l, j * downscale + k] = True

    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)

    mask_at_box = mask_at_box.reshape(H, W)
    full_mask = mask & mask_at_box  # get full mask, including mask_at_box and mask, still (512 * 512)
    box_mask = mask[mask_at_box]  # get valid indices in mask_at_box (mask_size,)
    mask_at_box = mask_at_box[mask]  # get valid mask_at_box value in (128, 128)
    near = near[box_mask].astype(np.float32)
    far = far[box_mask].astype(np.float32)
    ray_o = ray_o[full_mask].astype(np.float32)
    ray_d = ray_d[full_mask].astype(np.float32)

    return ray_o, ray_d, near, far, full_mask, mask_at_box


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat.astype(np.float32)


def get_rigid_transformation_and_joints(poses, joints, parents):
    """
    poses: n_bones x 3
    joints: n_bones x 3
    parents: n_bones
    """
    n_bones = len(joints)
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    # first rotate then transform
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([n_bones, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    # but this is a world transformation, with displacement...?
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):  # assuming parents are in topological order
        # curr_res = np.dot(transforms_mat[i], transform_chain[parents[i]])
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])  # !: THEY'RE RIGHT, LEARN FORWARD KINEMATICS
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    # !: AND THIS WEIRD STUFF IS TRYING TO MOVE VERTEX FROM VERTEX COORDINATES TO JOINT COORDINATES
    # !: AND THIS IS THE CORRECT IMPLEMENTATION...

    # !: THIS IS JUST TOO CLEVER...
    # These three lines is effectively doing: transforms = transforms * (negative trarslation matrix for all joints)
    joints_vector = np.concatenate([joints, np.zeros([n_bones, 1])], axis=1)
    rot_joints = np.sum(transforms * joints_vector[:, None], axis=2)  # This is effectively matmul
    transforms[..., 3] = transforms[..., 3] - rot_joints  # add in the translation, we should translate first

    joints_points = np.concatenate([joints, np.ones([n_bones, 1])], axis=1)
    pose_joints = np.sum(transforms * joints_points[:, None], axis=2)  # This is effectively matmul

    transforms = transforms.astype(np.float32)
    return transforms, pose_joints[:, :3]


def get_rigid_transformation(poses, joints, parents):
    """
    poses: n_bones x 3
    joints: n_bones x 3
    parents: n_bones
    """
    transforms = get_rigid_transformation_and_joints(poses, joints, parents)[0]
    return transforms


def padding_bbox_HW(bbox, h, w):
    padding = 10
    bbox[0] = bbox[0] - 10
    bbox[1] = bbox[1] + 10

    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    # a magic number of pytorch3d
    ratio = 1.5

    if height / width > ratio:
        min_size = int(height / ratio)
        if width < min_size:
            padding = (min_size - width) // 2
            bbox[0, 0] = bbox[0, 0] - padding
            bbox[1, 0] = bbox[1, 0] + padding

    if width / height > ratio:
        min_size = int(width / ratio)
        if height < min_size:
            padding = (min_size - height) // 2
            bbox[0, 1] = bbox[0, 1] - padding
            bbox[1, 1] = bbox[1, 1] + padding

    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def padding_bbox(bbox, img):
    return padding_bbox_HW(bbox, *img.shape[:2])


def get_crop_box(H, W, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox_HW(bbox, H, W)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return K, bbox


def crop_image_msk(img, msk, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox(bbox, img)

    crop = img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    crop_msk = msk[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

    # calculate the shape
    shape = crop.shape
    x = 8
    height = (crop.shape[0] | (x - 1)) + 1
    width = (crop.shape[1] | (x - 1)) + 1

    # align image
    aligned_image = np.zeros([height, width, 3])
    aligned_image[:shape[0], :shape[1]] = crop
    aligned_image = aligned_image.astype(np.float32)

    # align mask
    aligned_msk = np.zeros([height, width])
    aligned_msk[:shape[0], :shape[1]] = crop_msk
    aligned_msk = (aligned_msk == 1).astype(np.uint8)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return aligned_image, aligned_msk, K, bbox


def random_crop_image(img, msk, K, min_size, max_size):
    # sometimes we sample regions with no valid pixel at all, this can be problematic for the training loop
    # there's an assumption that the `msk` is always inside `mask_at_box`
    # thus, if we're sampling inside the `msk`, we'll always be getting the correct results
    H, W = img.shape[:2]
    min_HW = min(H, W)
    min_HW = min(min_HW, max_size)

    max_size = min_HW
    # min_size = int(min(min_size, 0.8 * min_HW))
    if max_size < min_size:
        H_size = np.random.randint(min_size, max_size)
    else:
        H_size = min_size

    W_size = H_size
    x = 8
    H_size = (H_size | (x - 1)) + 1
    W_size = (W_size | (x - 1)) + 1

    # randomly select begin_x and begin_y
    coord = np.argwhere(msk == 1)
    center_xy = coord[np.random.randint(0, len(coord))][[1, 0]]
    min_x, min_y = center_xy[0] - W_size // 2, center_xy[1] - H_size // 2
    max_x, max_y = min_x + W_size, min_y + H_size
    if min_x < 0:
        min_x, max_x = 0, W_size
    if max_x > W:
        min_x, max_x = W - W_size, W
    if min_y < 0:
        min_y, max_y = 0, H_size
    if max_y > H:
        min_y, max_y = H - H_size, H

    # crop image and mask
    begin_x, begin_y = min_x, min_y
    img = img[begin_y:begin_y + H_size, begin_x:begin_x + W_size]
    msk = msk[begin_y:begin_y + H_size, begin_x:begin_x + W_size]

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - begin_x
    K[1, 2] = K[1, 2] - begin_y
    K = K.astype(np.float32)

    return img, msk, K


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_bounds(xyz, box_padding=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= box_padding
    max_xyz += box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds


def crop_mask_edge(msk):
    msk = msk.copy()
    border = 10
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = 100
    return msk


def adjust_hsv(img, saturation, brightness, contrast):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * saturation
    hsv[..., 1] = np.minimum(hsv[..., 1], 255)
    hsv[..., 2] = hsv[..., 2] * brightness
    hsv[..., 2] = np.minimum(hsv[..., 2], 255)
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32) * contrast
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img
