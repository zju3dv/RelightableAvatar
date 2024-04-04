# Typing
from typing import Callable, List, Tuple

# Torch
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Others
import numpy as np
from tqdm import tqdm

# Utils
from lib.utils.log_utils import log
from lib.utils.base_utils import dotdict
from lib.utils.net_utils import linear_gather, multi_gather, multi_gather_tris, normalize_sum, batch_aware_indexing, multi_scatter, multi_scatter_

# PyTorch3D
from pytorch3d import _C
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points, sample_farthest_points
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.loss.point_mesh_distance import _DEFAULT_MIN_TRIANGLE_AREA
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes, _rand_barycentric_coords

from functools import lru_cache


def caching_lbvh_constructor(
    # These are required
    p1: torch.Tensor,
    return_sorted: bool = True,
    leaf_size=32,
    compact=False,
    shrink_to_fit=False,

    # These are dynamic
    K: int = 1,
    radius=None,
):
    key = (p1.data_ptr(), return_sorted, leaf_size, compact, shrink_to_fit)
    if key not in caching_lbvh_constructor.cache:
        import cupy
        import cupy_knn
        p1_cupy = cupy.asarray(p1)
        lbvh = cupy_knn.LBVHIndex(leaf_size=leaf_size,
                                  compact=compact,
                                  shrink_to_fit=shrink_to_fit,
                                  sort_queries=return_sorted)
        lbvh.build(p1_cupy)
        caching_lbvh_constructor.cache[key] = lbvh

        if len(caching_lbvh_constructor.cache) > caching_lbvh_constructor.maxsize:
            # caching_lbvh_constructor.cache.popitem(last=False)
            # Updated answer
            # In the context of the question, we are dealing with pseudocode, but starting in Python 3.8, := is actually a valid operator that allows for assignment of variables within expressions:
            # https://stackoverflow.com/questions/26000198/what-does-colon-equal-in-python-mean
            (k := next(iter(caching_lbvh_constructor.cache)), caching_lbvh_constructor.cache.pop(k))
    else:
        lbvh = caching_lbvh_constructor.cache[key]

    lbvh.prepare_knn_default(K, radius=radius)
    return lbvh


caching_lbvh_constructor.maxsize = 128
caching_lbvh_constructor.cache = dotdict()


def cupy_knn_points(p1: torch.Tensor,
                    p2: torch.Tensor,
                    K: int = 1,
                    return_nn: bool = True,
                    return_sorted: bool = True,

                    leaf_size=32,
                    compact=False,
                    shrink_to_fit=False,
                    radius=None,

                    return_lbvh=False,
                    ):
    import cupy
    import cupy_knn
    # !: BATCH
    lbvh = caching_lbvh_constructor(p1, leaf_size, compact, shrink_to_fit, return_sorted, K, radius)

    p2_cupy = cupy.asarray(p2)
    idx_cupy, d2_cupy, nn_cupy = lbvh.query_knn(p2_cupy)
    idx_tensor = torch.as_tensor(idx_cupy.astype(cupy.int64), device=p1.device)[None]  # !: BATCH
    d2_tensor = torch.as_tensor(d2_cupy, device=p1.device)[None]
    nn_tensor = torch.as_tensor(nn_cupy.astype(cupy.int64), device=p1.device)[None]

    if not return_lbvh:
        return d2_tensor, idx_tensor, nn_tensor
    else:
        return d2_tensor, idx_tensor, nn_tensor, return_lbvh


def geodesic_knn(pts: torch.Tensor,
                 verts: torch.Tensor,
                 norm: torch.Tensor,
                 tverts: torch.Tensor,
                 tnorm: torch.Tensor,
                 K: int,
                 th: float,
                 use_geodesic_filter: bool = True,
                 ):
    """
    will find distances for all points
    will filter points based on distance (i.e. larger than th, no index, only sdf)
    """
    if not use_geodesic_filter: return knn_with_filter(pts, verts, norm, K, th)
    # log('Using geodesic KNN')
    B, P, _ = pts.shape
    B, N, _ = verts.shape

    # find cloest distances from points of the smpl mesh
    d2, nn, _ = knn_points(pts, verts, K=K, return_nn=False, return_sorted=True)  # B, P, K; all valid index, not all valid points
    # selecting the first of the cloeset K points for normal computation
    dist_batch = d2.sqrt()  # B, P, K
    norm_batch = multi_gather(norm[..., None, :].expand(B, N, K, 3), nn, dim=1)  # B, P, K, 3
    verts_batch = multi_gather(verts[..., None, :].expand(B, N, K, 3), nn, dim=1)  # B, P, K, 3
    dot_batch = ((pts[..., None, :] - verts_batch) * norm_batch).sum(dim=-1)  # B, P, K, 1
    sdf_batch = dist_batch * dot_batch.sign()  # B, P, K, to match d2 return shape

    # define filter for valid points (distance to smpl mesh within a certain threshold)
    nn_batch = nn  # indices as knn results
    d2_min = d2[..., 0]  # B, P, smallest distance (returned value of KNN is always sorted (?)
    d2_min, inds, S = batch_aware_indexing(d2_min < th ** 2, -d2_min)  # MARK: SYNC

    d2 = multi_gather(d2, inds)  # B, S, K
    nn = multi_gather(nn, inds)  # B, S, K
    ppts = multi_gather(pts, inds)  # B, S, 3

    # Now, filter return values of KNN with geodesic distance defined on the canonical model
    # pts: B, P, 3
    # sdf_batch: B, P, K, to be filled
    # nn_batch: B, P, K, to be filled
    # inds: B, S
    # S: scalar
    # d2: B, S, K, to be filled
    # nn: B, S, K, to be filled
    # ppts: B, S, 3
    tv = multi_gather(tverts[:, None].expand(B, S, N, -1), nn)  # B, S, K, 3
    tv_cls = tv[..., :1, :]
    tv_to_tv_cls = (tv - tv_cls).pow(2).sum(-1)
    msk = tv_to_tv_cls < th ** 2
    d2 = torch.where(msk, d2, d2[..., :1])  # fill invalid regions with closest
    nn = torch.where(msk, nn, nn[..., :1])

    tv = multi_gather(tverts[:, None].expand(B, P, N, -1), nn_batch)  # B, P, K, 3
    tv_cls = tv[..., :1, :]
    tv_to_tv_cls = (tv - tv_cls).pow(2).sum(-1)
    msk = tv_to_tv_cls < th ** 2
    sdf_batch = torch.where(msk, sdf_batch, sdf_batch[..., :1])
    nn_batch = torch.where(msk, nn_batch, nn_batch[..., :1])

    return sdf_batch, nn_batch, inds, S, d2, nn, ppts


def knn_with_filter(pts: torch.Tensor,
                    verts: torch.Tensor,
                    norm: torch.Tensor,
                    K: int,
                    th: float,
                    ):
    """
    will find distances for all points
    will filter points based on distance (i.e. larger than th, no index, only sdf)
    """
    B, P, _ = pts.shape
    B, N, _ = verts.shape

    # find cloest distances from points of the smpl mesh
    d2, inds, _ = knn_points(pts, verts, K=K, return_nn=False, return_sorted=False)  # B, P, K; all valid index, not all valid points
    # selecting the first of the cloeset K points for normal computation (since we don't need to compute normals for close to surface points)
    dist_batch = d2.mean(dim=-1, keepdim=True).sqrt()  # B, P, 1 # to make it a bit smoother
    norm_batch = multi_gather(norm[..., None, :].expand(B, N, K, 3), inds, dim=1)  # B, P, K, 3
    verts_batch = multi_gather(verts[..., None, :].expand(B, N, K, 3), inds, dim=1)  # B, P, K, 3
    dot_batch = ((pts[..., None, :] - verts_batch) * norm_batch).sum(dim=-1, keepdim=True)  # B, P, K, 1
    sdf_batch = dist_batch * dot_batch.sign().max(dim=-2)[0]  # B, P, 1

    # define filter for valid points (distance to smpl mesh within a certain threshold)
    d2_batch = d2[..., 0]  # B, P, smallest distance (returned value of KNN is always sorted (?)
    d2_batch, inds_batch, S = batch_aware_indexing(d2_batch < th ** 2, -d2_batch)  # MARK: SYNC

    d2 = multi_gather(d2, inds_batch)  # B, S, K
    inds = multi_gather(inds, inds_batch)  # B, S, K
    pts = multi_gather(pts, inds_batch)  # B, S, 3

    return sdf_batch, inds, inds_batch, S, d2, inds, pts


class BatchPointMeshDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the 
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists.sqrt(), idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


# pyre-fixme[16]: `_PointFaceDistance` has no attribute `apply`.
batch_point_mesh_distance = BatchPointMeshDistance.apply


class PointMeshDistance(Function):
    # PointFaceDistance
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    RETURNS FACE_IDX AND ACTUAL DISTANCE INSTEAD OF SQUARED
    """
    @staticmethod
    def forward(ctx, points, tris, n_batch=1):
        """
        Args:
            points: FloatTensor of shape `(P, 3)`
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            n_batch: Num of batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the REAL! NOT SQUARED
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.
        """
        n_points = points.shape[0]
        device = points.device
        zeros = torch.zeros(n_batch, dtype=torch.long, device=device)
        dists, idxs = _C.point_face_dist_forward(
            points, zeros, tris, zeros, n_points, _DEFAULT_MIN_TRIANGLE_AREA
        )
        ctx.save_for_backward(points, tris, idxs)
        return dists.sqrt(), idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, _DEFAULT_MIN_TRIANGLE_AREA
        )
        return grad_points, None, grad_tris, None, None


# pyre-fixme[16]: `_PointFaceDistance` has no attribute `apply`.
point_mesh_distance = PointMeshDistance.apply


def random_points_on_meshes_with_face_and_bary(meshes: Meshes, num_samples: int):
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    bary = torch.stack((w0, w1, w2), dim=2)
    return samples, sample_face_idxs, bary


def get_voxel_grid_and_update_bounds(voxel_size: List, bounds: torch.Tensor):
    # now here's the problem
    # 1. if you want the voxel size to be accurate, you bounds need to be changed along with this sampling process
    #    since the grid_sample will treat the bounds based on align_corners=True or not
    #    say we align corners, the actual bound on the sampled tpose blend weight should be determined by the actual sampling voxels
    #    not the bound that we kind of used to produce the voxels, THEY DO NOT LINE UP UNLESS your bounds is divisible by the voxel size in every direction

    # voxel_size: [0.005, 0.005, 0.005]
    # bounds: n_batch, 2, 3, initial bounds
    ret = []
    for b in bounds:
        x = torch.arange(b[0, 0].item(), b[1, 0].item() + voxel_size[0] / 2, voxel_size[0], dtype=b.dtype, device=b.device)
        y = torch.arange(b[0, 1].item(), b[1, 1].item() + voxel_size[1] / 2, voxel_size[1], dtype=b.dtype, device=b.device)
        z = torch.arange(b[0, 2].item(), b[1, 2].item() + voxel_size[2] / 2, voxel_size[2], dtype=b.dtype, device=b.device)
        pts = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        ret.append(pts)
    pts = torch.stack(ret)  # dim 0
    bounds = torch.stack([pts[:, 0, 0, 0], pts[:, -1, -1, -1]], dim=1)  # dim 1 n_batch, 2, 3
    return pts, bounds


def optimize_until_no_nan(func: Callable[..., torch.Tensor], *args: torch.Tensor):
    # Note: assuming first value in args is to be optimized and can be NaN
    # FIXME: nasty fix for nan: repeating until it's not a nan anymore, while True is bad...
    param_to_optim = args[0]
    while True:

        param_to_optim = func(
            param_to_optim.clone(), *args[1:]
        )

        if param_to_optim.isnan().any():
            log("nan detected, repeating grid optimization step...", 'red')
            continue
        else:
            break
    return param_to_optim


def optimze_samples_from_volume(grid: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, bounds: torch.Tensor, n_free_pts: int = 1024, n_surf_pts: int = 1024, norm_expand_step: int = 2, norm_expand_factor: float = 0.02, n_iter: int = 128, lr: float = 1e-3, clip_grad=10):

    # post_processing: apply additional constraints on returned values (like sums to one)
    n_batch, w, h, d, D = grid.shape
    diag = bounds[:, 1] - bounds[:, 0]  # B, 3

    mesh = Meshes(verts, faces)
    norm = mesh.verts_normals_padded()

    grid.requires_grad = True
    optim = Adam([grid], lr=lr)

    p = tqdm(range(n_iter))
    with torch.enable_grad():  # you've turned off grad before main
        for _ in p:

            free_pts = torch.rand([n_batch, n_free_pts, 3], dtype=grid.dtype, device=grid.device)
            free_pts *= diag  # [0,1] -> diagonal
            free_pts += bounds[:, 0]  # diagonal -> shifted
            surf, norm = sample_points_from_meshes(mesh, n_surf_pts, True)
            surf_pts = torch.cat([surf + norm * norm_expand_factor * (torch.rand(1, dtype=grid.dtype, device=grid.device) * 2 - 1) for _ in range(norm_expand_step)], dim=1)
            pts = torch.cat([free_pts, verts, surf_pts], dim=1)  # make sure verts gets mapped correctly

            pred = sample_grid(pts, grid, bounds)
            gt, dists = sample_closest_points_on_surface(pts, verts, faces, values)

            loss = ((pred - gt)**2).sum(dim=-1).mean()
            p.set_description(f"l2: {loss:.6f}")

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_([grid], clip_grad)
            optim.step()

    return grid


def batch_points_to_barycentric(points: torch.Tensor, triangles: torch.Tensor, method="cross", eps: float = 1e-16):
    return points_to_barycentric(points.view(-1, 3), triangles.view(-1, 3, 3), method, eps).view(*points.shape)


def points_to_barycentric(points: torch.Tensor, triangles: torch.Tensor, method="cross", eps: float = 1e-16):
    # points: n_points, 3
    # triangles: n_points, 3, 3

    def method_cross(edge_vectors: torch.Tensor, w: torch.Tensor):
        n = torch.cross(edge_vectors[:, 0], edge_vectors[:, 1])
        denominator = torch.bmm(n[:, None], n[..., None])[:, 0, 0]
        denominator[denominator.abs() < eps] = eps

        barycentric = torch.zeros((len(triangles), 3), dtype=points.dtype, device=points.device)
        barycentric[:, 2] = torch.bmm(torch.cross(edge_vectors[:, 0], w)[:, None], n[..., None])[:, 0, 0] / denominator  # u
        barycentric[:, 1] = torch.bmm(torch.cross(w, edge_vectors[:, 1])[:, None], n[..., None])[:, 0, 0] / denominator  # v
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]  # w
        return barycentric

    def method_cramer(edge_vectors: torch.Tensor, w: torch.Tensor):
        # n_points, 1, 3 @ n_points, 3, 1 -> n_points, 1, 1
        dot00 = torch.bmm(edge_vectors[:, 0][:, None], edge_vectors[:, 0][..., None])[:, 0, 0]  # n_points
        dot01 = torch.bmm(edge_vectors[:, 0][:, None], edge_vectors[:, 1][..., None])[:, 0, 0]
        dot02 = torch.bmm(edge_vectors[:, 0][:, None], w[..., None])[:, 0, 0]
        dot11 = torch.bmm(edge_vectors[:, 1][:, None], edge_vectors[:, 1][..., None])[:, 0, 0]
        dot12 = torch.bmm(edge_vectors[:, 1][:, None], w[..., None])[:, 0, 0]

        denominator = dot00 * dot11 - dot01 * dot01
        denominator[denominator.abs() < eps] = eps

        barycentric = torch.zeros((len(triangles), 3), dtype=points.dtype, device=points.device)
        barycentric[:, 2] = (dot00 * dot12 - dot01 * dot02) / denominator  # u
        barycentric[:, 1] = (dot11 * dot02 - dot01 * dot12) / denominator  # v
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]  # w
        return barycentric

    def constraint_barycentric(barycentric: torch.Tensor):
        barycentric[barycentric < 0] = 0
        barycentric = normalize_sum(barycentric)
        return barycentric

    edge_vectors = triangles[:, 1:] - triangles[:, :1]  # n_points, 2, 3
    w = points - triangles[:, 0].view(-1, 3)  # n_points, 3

    # trimesh.triangles.points_to_barycentric(triangles.cpu().numpy(), points.cpu().numpy())
    if method == "cramer":
        barycentric = method_cramer(edge_vectors, w)  # might be out of bound
    else:
        barycentric = method_cross(edge_vectors, w)  # might be out of bound

    barycentric = constraint_barycentric(barycentric)
    return barycentric


def sample_texture(uvs: torch.Tensor, textures: torch.Tensor, from_ndc: bool = False, flip_tex: bool = True) -> torch.Tensor:
    """sample rgb values (or not) on textures, note that this is not differentiable w.r.t. uvs
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    uvs: n_batch, n_points, 3
    textures: n_batch, h, w, 3

    returns:
    values: n_batch, n_points, 3
    """
    # interpolate blend weights
    if not from_ndc:
        grid_coords = uvs * 2 - 1  # to ndc space, n_batch, n_points, 3
    else:
        grid_coords = uvs

    values = textures.permute(0, 3, 1, 2)  # n_batch, 3, h, w

    if flip_tex:
        values = values.flip(2)  # TODO: why flip the texture image H before sampling?
    grid_coords = grid_coords[:, None]

    values = grid_sample(values,  # n_batch, 3, h, w
                         grid_coords,  # n_batch, 1, n_points, 2
                         padding_mode='border',
                         align_corners=False)  # n_batch, 3, h, w
    values = values[:, :, 0].permute(0, 2, 1)  # (n_batch, 3, n_points) -> (n_batch, n_points, 3)

    return values


def sample_grid(pts: torch.Tensor, values: torch.Tensor, bounds: torch.Tensor):
    """sample blend weights for points
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    pts: n_batch, n_points, 3
    values: n_batch, w, h, d, n_bones
    bounds: n_batch, 2, 3

    returns:
    values: n_batch, n_points, n_bones
    """
    # interpolate blend weights
    diagonal = bounds[:, 1:] - bounds[:, :1]  # n_batch, 1, 3
    grid_coords = (pts - bounds[:, :1]) / diagonal  # n_batch, n_points, 3
    grid_coords = grid_coords * 2 - 1  # to ndc space, n_batch, n_points, 3
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    # TODO: what the heck between the dhw and whd conversion?
    values = values.permute(0, 4, 1, 2, 3)  # n_batch, n_bones, w, h, d
    grid_coords = grid_coords[:, None, None]

    values = grid_sample(values,  # n_batch, n_bones, w, h, d
                         grid_coords,  # n_batch, 1, 1, n_points, 3 (now indexing zyx)
                         padding_mode='border',
                         align_corners=False)  # n_batch, n_bones, w, h, d
    values = values[:, :, 0, 0].permute(0, 2, 1)  # (n_batch, n_bones, n_points) -> (n_batch, n_points, n_bones)

    return values


def expand_points_for_sampling(verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, expand_range: List, expand_step: int):
    # prepare emission (along normal direction)
    # mesh = get_mesh(verts, faces)
    mesh = Meshes(verts, faces)
    vert_cnt = verts.shape[1]  # consider batch dimension
    vert_norms = mesh.verts_normals_padded()
    expand_len = expand_range[1] - expand_range[0]
    expand_min = expand_range[0]
    expand_cnt = expand_step + 1
    # vert_norms = torch.tensor(mesh.vertex_normals, dtype=pts.dtype, device=pts.device)[None] # add batch dimension
    expand_verts = torch.cat(
        [
            (expand_min + i * expand_len / expand_step) * vert_norms + verts for i in range(expand_cnt)
        ],
        dim=1  # verts has a batch dimension
    )
    expand_faces = torch.cat(
        [
            faces + i * vert_cnt for i in range(expand_cnt)
        ],
        dim=1,  # faces has a batch dimension
    )
    expand_values = values.repeat(1, expand_cnt, 1)

    return expand_verts, expand_faces, expand_values


def expand_verts_faces_along_normal(verts: torch.Tensor, faces: torch.Tensor, expand_range: List = [0.05, -0.05], expand_step: int = 5, randomize: bool = True):
    # NOTE: this will include the range as open (unlike get_wsampling_pts implementation)
    if expand_step == 1:
        return verts, faces

    B, V, _ = verts.shape
    mesh = Meshes(verts, faces)
    vert_norms = mesh.verts_normals_padded()
    expand_len = expand_range[1] - expand_range[0]
    expand_min = expand_range[0]

    expanded_height: torch.Tensor = torch.arange(expand_step, device=verts.device, dtype=verts.dtype)
    if randomize:
        expanded_height = expanded_height + torch.rand(expand_step, device=verts.device, dtype=verts.dtype)
    expanded_height = expanded_height * expand_len / (expand_step - 1) + expand_min
    expanded_height = expanded_height[:, None, None].expand(-1, V, -1)

    expanded_verts = verts.expand(expand_step, -1, -1) + vert_norms.expand(expand_step, -1, -1) * expanded_height
    expanded_faces = faces.expand(expand_step, -1, -1)
    return expanded_verts, expanded_faces, expanded_height


def sample_grid_closest_points(pts: torch.Tensor, verts: torch.Tensor, values: torch.Tensor):
    return [x.view(*pts.shape[:-1], -1) for x in sample_closest_points(pts.view(pts.shape[0], -1, 3), verts, values)]  # (1, whd, n_bones+3)


def sample_grid_closest_points_on_surface(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor):
    return [x.view(*pts.shape[:-1], -1) for x in sample_closest_points_on_surface(pts.view(pts.shape[0], -1, 3), verts, faces, values)]  # (1, whd, n_bones+3)


def sample_grid_closest_points_on_expanded_surface(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, expand_range: List, expand_step: int):
    expand_verts, expand_faces, expand_values = expand_points_for_sampling(verts, faces, values, expand_range, expand_step)
    return sample_grid_closest_points_on_surface(pts, expand_verts, expand_faces, expand_values)


def cast_knn_points(src, ref, K=1):
    ret = knn_points(src.float(), ref.float(), K=K, return_nn=False, return_sorted=False)
    dists, idx = ret.dists, ret.idx  # returns l2 distance?
    ret = dotdict()
    ret.dists = dists.sqrt()
    ret.idx = idx
    return ret


def sample_closest_points(src: torch.Tensor, ref: torch.Tensor, values: torch.Tensor = None):
    n_batch, n_points, _ = src.shape
    ret = cast_knn_points(src, ref, K=1)  # (n_batch, n_points, K)
    dists, vert_ids = ret.dists, ret.idx
    if values is None:
        return dists.view(n_batch, n_points, 1)
    values = values.view(-1, values.shape[-1])  # (n, D)
    sampled = values[vert_ids]  # (s, D)
    return sampled.view(n_batch, n_points, -1), dists.view(n_batch, n_points, 1)


def sample_K_closest_points(src: torch.Tensor, ref: torch.Tensor, K: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    # not so useful to aggregate all K points
    ret = cast_knn_points(src, ref, K=K)
    return ret.dists, ret.idx  # (n_batch, n_points, K)


def sample_blend_K_closest_points(src: torch.Tensor, ref: torch.Tensor, values: torch.Tensor = None, K: int = 4, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    # not so useful to aggregate all K points
    n_batch, n_points, _ = src.shape
    ret = cast_knn_points(src, ref, K=K)
    dists, vert_ids = ret.dists, ret.idx  # (n_batch, n_points, K)
    # sampled = values[vert_ids]  # (n_batch, n_points, K, D)
    weights = 1 / (dists + eps)
    weights /= weights.sum(dim=-1, keepdim=True)
    dists = torch.einsum('ijk,ijk->ij', dists, weights)
    if values is None:
        return dists.view(n_batch, n_points, 1)
    # sampled *= weights[..., None]  # augment weight in last dim for bones # written separatedly to avoid OOM
    # sampled = sampled.sum(dim=-2)  # sum over second to last for weighted bw
    values = values.view(-1, values.shape[-1])  # (n, D)
    sampled = torch.einsum('ijkl,ijk->ijl', values[vert_ids], weights)
    return sampled.view(n_batch, n_points, -1), dists.view(n_batch, n_points, 1)


def sample_closest_points_on_surface_approx(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, n_surf_pts: int = 16384):
    # even slower...
    # expect all to have batch dim
    n_batch, n_points, _ = points.shape

    # similiar to computing point mesh distance and then do a barycentric
    # but directly sample points on surface, then do a ball query for values
    mesh = Meshes(verts, faces)
    # FIXME: bary is not the same bary as in points_to_barycentric
    # surf: (b, n, 3)
    # face_ids: (b, n)
    # bary: (b, n, 3)
    surf, face_ids, bary = random_points_on_meshes_with_face_and_bary(mesh, n_surf_pts)
    # dists: (b, n, 1)
    # surf_ids: (b, n, 1)
    # nn: (b, n, 1, 3)
    dists, vert_ids = cast_knn_points(points, surf, K=1)

    values = values.view(-1, values.shape[-1])  # (n, D)
    faces = faces.view(-1, faces.shape[-1])  # (n ,3)
    bary = bary.view(-1, bary.shape[-1])  # (n, 3)
    face_ids = face_ids.view(-1)  # (f)

    sampled = torch.sum(values[faces[face_ids]] *  # (n, 3, 3)
                        bary[..., None],  # (n, 3, 1)
                        dim=1)

    vert_ids = vert_ids.view(-1)  # (s)
    sampled = sampled[vert_ids]  # (s, D)
    return sampled.view(n_batch, n_points, sampled.shape[-1]), dists.view(n_batch, n_points, 1)


def sample_closest_points_on_surface(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor = None):
    # samples values by barycentricing the closest points on the mesh
    # points: n_batch, n_points, 3
    # verts: n_batch, n_verts, 3
    # faces: n_batch, n_faces, 3 (long)
    # values: n_batch, n_verts, D (might need to augment some)

    # FIXME: this is not gonna be differentiable
    # https://github.com/facebookresearch/pytorch3d/issues/193, pytorch doesn't expose the indices' api
    # and we decide to resort back to Trimesh for some R-trees
    # this goes inefficient, and maybe we should not calculate this on the fly?
    # TODO: test out the performance of this
    n_batch, n_points, _ = points.shape

    points = points.view(-1, 3)
    verts = verts.view(-1, 3)
    faces = faces.view(-1, 3)

    # # FIXME: trimesh functions returns nothing in the eye of pylance
    # # using trimesh to access them to make pylance happy
    # import trimesh
    # mesh = trimesh.Trimesh(verts, faces)
    # closest, distance, face_id = trimesh.proximity.closest_point(mesh, points)
    # closest_verts = faces[face_id]  # (n, 3, 3)
    # barycentric = trimesh.triangles.points_to_barycentric(verts[closest_verts], closest)  # (n, 3)

    # device = points.device
    # closest_verts = torch.tensor(closest_verts).to(device)
    # barycentric = torch.tensor(barycentric).to(device)
    # values = torch.sum(values[closest_verts] *  # (n, 3, 3)
    #                    barycentric[..., None],  # (n, 3, 1)
    #                    dim=1)
    # we use pytorch for a faster cuda version instead of implementing by hand
    dists, face_ids = point_mesh_distance(points, multi_gather_tris(verts, faces), n_batch)
    if values is None:
        return dists.view(n_batch, n_points, 1)
    else:
        values = values.view(-1, values.shape[-1])
    bary = points_to_barycentric(points, verts[faces[face_ids]])  # (n, 3)
    interp = torch.sum(values[faces[face_ids]] *  # (n, 3, 3)
                       bary[..., None],  # (n, 3, 1)
                       dim=1)

    return interp.view(n_batch, n_points, -1), dists.view(n_batch, n_points, 1)


def grid_sample(input: torch.Tensor, grid: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # https://github.com/pytorch/pytorch/issues/34704
    # RuntimeError: derivative for grid_sampler_2d_backward is not implemented
    # this implementation might be slower than the cuda one
    if args or kwargs:
        # warnings.warn(message=f'unused arguments for grid_sample: {args}, {kwargs}')
        return F.grid_sample(input, grid, *args, **kwargs)
    if input.ndim == 4:
        # invoke 2d custom grid_sampling
        assert grid.ndim == 4, '4d input needs a 4d grid'
        return grid_sample_2d(input, grid)
    elif input.ndim == 5:
        # invoke 3d custom grid_sampling
        assert grid.ndim == 5, '5d input needs a 5d grid'
        return grid_sample_3d(input, grid)
    else:
        raise NotImplementedError(f'grid_sample not implemented for {input.ndim}d input')


def grid_sample_2d(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():

        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val
