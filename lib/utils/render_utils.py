import os
import cv2
import json
import numpy as np

from scipy import interpolate
from lib.config import cfg
from lib.utils.data_utils import get_rays, get_near_far


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def load_cam(ann_file):
    if ann_file.endswith('.json'):
        annots = json.load(open(ann_file, 'r'))
        cams = annots['cams']['20190823']
    else:
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']

    K = []
    RT = []
    lower_row = np.array([[0., 0., 0., 1.]])

    for i in range(len(cams['K'])):
        K.append(np.array(cams['K'][i]))
        K[i][:2] = K[i][:2] * cfg.ratio

        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i]) / 1000.
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))

    return np.array(K).astype(np.float32), np.array(RT).astype(np.float32)


def get_center_rayd(K, RT):
    H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
    RT = np.array(RT)
    ray_o, ray_d = get_rays(H, W, K, RT[:3, :3], RT[:3, 3])
    return ray_d[H // 2, W // 2]


def get_interpolate_params(exts, smoothing_term=1.0):
    """Return B-spline interpolation parameters for the camera
    TODO: Actually this should be implemented as a general interpolation function
    Reference get_camera_up_front_center for the definition of worldup, front, center
    Args:
        smoothing_term(float): degree of smoothing to apply on the camera path interpolation
    """
    # - R^t @ T = cam2world translation
    # TODO: load from cam_points to avoid repeated computation
    all_cens = -np.einsum("bij,bj->bi", exts[:, :3, :3].transpose(0, 2, 1), exts[:, :3, 3]).T
    all_fros = exts[:, 2, :3].T  # (3, 21)
    all_wups = -exts[:, 1, :3].T  # (3, 21)
    cen_tck, cen_u = interpolate.splprep(all_cens, s=smoothing_term, per=0)  # array of u corresponds to parameters of specific camera points
    fro_tck, fro_u = interpolate.splprep(all_fros, s=smoothing_term, per=0)  # array of u corresponds to parameters of specific camera points
    wup_tck, wup_u = interpolate.splprep(all_wups, s=smoothing_term, per=0)  # array of u corresponds to parameters of specific camera points
    return cen_tck, cen_u, fro_tck, fro_u, wup_tck, wup_u


def flatten_inner_dim(x):
    # only a incomplete utility function
    # doesn't check that outer dim is there
    if isinstance(x, list):
        if len(x) > 1:
            return [flatten_inner_dim(v) for v in x]
        else:
            return flatten_inner_dim(x[0])
    elif isinstance(x, np.ndarray):
        x = x.squeeze()  # no recursion needed for np array
        if x.size > 1:
            return np.array([flatten_inner_dim(v) for v in x])
        else:
            return x.item()


def interpolate_path(exts, smoothing_term=10.0):
    exts = np.array(exts)
    if len(exts) == 1:  # only one view
        inter_w2c = exts.repeat(cfg.num_render_view, 0)
        return inter_w2c
    inter_w2c = []
    cen_tck, cen_u, fro_tck, fro_u, wup_tck, wup_u = get_interpolate_params(exts, smoothing_term)
    for i in range(cfg.num_render_view):
        u = i / cfg.num_render_view
        center = np.array(interpolate.splev(u, cen_tck))
        v_front = np.array(interpolate.splev(u, fro_tck))
        v_world_up = np.array(interpolate.splev(u, wup_tck))
        v_right = np.cross(v_front, v_world_up)
        v_down = -v_world_up
        c2w = np.zeros((4, 4))
        c2w[-1, -1] = 1
        c2w[:3, 0] = normalize(v_right)
        c2w[:3, 1] = normalize(v_down)
        c2w[:3, 2] = normalize(v_front)
        c2w[:3, 3] = center
        w2c = np.linalg.inv(c2w)
        inter_w2c.append(w2c)
    return np.array(inter_w2c)


def gen_path(RT, center=[], z_off=-1) -> np.ndarray:
    if cfg.interpolate_path:
        return interpolate_path(RT, cfg.smoothing_term).astype(np.float32)

    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1], -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))

    if z_off < 0:
        if not len(center):
            z_off = 1.3
        else:
            z_off = 0.0

    if not len(center):
        center = RT[:, :3, 3].mean(0)
    else:
        center = np.array(center)

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, cfg.num_render_view + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world - np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return np.array(render_w2c).astype(np.float32)


def read_voxel(frame, args):
    voxel_path = os.path.join(args['data_root'], 'voxel', args['human'],
                              '{}.npz'.format(frame))
    voxel_data = np.load(voxel_path)
    occupancy = np.unpackbits(voxel_data['compressed_occupancies'])
    occupancy = occupancy.reshape(cfg.res, cfg.res,
                                  cfg.res).astype(np.float32)
    bounds = voxel_data['bounds'].astype(np.float32)
    return occupancy, bounds


def image_rays(RT, K, bounds):
    H = cfg.H * cfg.ratio
    W = cfg.W * cfg.ratio
    ray_o, ray_d = get_rays(H, W, K, RT[:3, :3], RT[:3, 3])

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])

    return ray_o, ray_d, near, far, center, scale, mask_at_box


def get_image_rays0(RT0, RT, K, bounds):
    """
    Use RT to get the mask_at_box and fill this region with rays emitted from view RT0
    """
    H = cfg.H * cfg.ratio
    ray_o, ray_d = get_rays(H, H, K, RT[:3, :3], RT[:3, 3])

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)

    ray_o, ray_d = get_rays(H, H, K, RT0[:3, :3], RT0[:3, 3])
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d[mask_at_box]

    return ray_d


def save_img(img, frame_root, index, mask_at_box):
    H = int(cfg.H * cfg.ratio)
    rgb_pred = img['rgb_map'][0].detach().cpu().numpy()
    mask_at_box = mask_at_box.reshape(H, H)

    img_pred = np.zeros((H, H, 3))
    img_pred[mask_at_box] = rgb_pred
    img_pred[:, :, [0, 1, 2]] = img_pred[:, :, [2, 1, 0]]

    print("saved frame %d" % index)
    cv2.imwrite(os.path.join(frame_root, '%d.jpg' % index), img_pred * 255)
