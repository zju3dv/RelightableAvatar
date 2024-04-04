import torch
import torch.nn.functional as F

from typing import List
from pytorch3d import transforms
from lib.utils.net_utils import batch_rodrigues
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_matrix
from lib.utils.sample_utils import sample_closest_points_on_surface, sample_closest_points, sample_blend_K_closest_points


def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)

# these works with an extra batch dimension
# Batched inverse of lower triangular matrices

# @torch.jit.script


def torch_trace(x: torch.Tensor):
    return x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


# @torch.jit.script
def torch_inverse_decomp(L: torch.Tensor, eps=1e-10):
    n = L.shape[-1]
    invL = torch.zeros_like(L)
    for j in range(0, n):
        invL[..., j, j] = 1.0 / (L[..., j, j] + eps)
        for i in range(j + 1, n):
            S = 0.0
            for k in range(i + 1):
                S = S - L[..., i, k] * invL[..., k, j].clone()
            invL[..., i, j] = S / (L[..., i, i] + eps)

    return invL


def torch_inverse_3x3_precompute(R: torch.Tensor, eps=torch.finfo(torch.float).eps):
    # B, N, 3, 3
    """
    a, b, c | m00, m01, m02
    d, e, f | m10, m11, m12
    g, h, i | m20, m21, m22
    """

    if not hasattr(torch_inverse_3x3_precompute, 'g_idx_i'):
        g_idx_i = torch.tensor(
            [
                [
                    [[1, 1], [2, 2]],
                    [[1, 1], [2, 2]],
                    [[1, 1], [2, 2]],
                ],
                [
                    [[0, 0], [2, 2]],
                    [[0, 0], [2, 2]],
                    [[0, 0], [2, 2]],
                ],
                [
                    [[0, 0], [1, 1]],
                    [[0, 0], [1, 1]],
                    [[0, 0], [1, 1]],
                ],
            ], device='cuda', dtype=torch.long)

        g_idx_j = torch.tensor(
            [
                [
                    [[1, 2], [1, 2]],
                    [[0, 2], [0, 2]],
                    [[0, 1], [0, 1]],
                ],
                [
                    [[1, 2], [1, 2]],
                    [[0, 2], [0, 2]],
                    [[0, 1], [0, 1]],
                ],
                [
                    [[1, 2], [1, 2]],
                    [[0, 2], [0, 2]],
                    [[0, 1], [0, 1]],
                ],
            ], device='cuda', dtype=torch.long)

        g_signs = torch.tensor([
            [+1, -1, +1],
            [-1, +1, -1],
            [+1, -1, +1],
        ], device='cuda', dtype=torch.long)

        torch_inverse_3x3_precompute.g_idx_i = g_idx_i
        torch_inverse_3x3_precompute.g_idx_j = g_idx_j
        torch_inverse_3x3_precompute.g_signs = g_signs

    g_idx_i = torch_inverse_3x3_precompute.g_idx_i
    g_idx_j = torch_inverse_3x3_precompute.g_idx_j
    g_signs = torch_inverse_3x3_precompute.g_signs

    B, N, _, _ = R.shape

    minors = R.new_zeros(B, N, 3, 3, 2, 2)
    idx_i = g_idx_i.to(R.device, non_blocking=True)  # almost never need to copy
    idx_j = g_idx_j.to(R.device, non_blocking=True)  # almost never need to copy
    signs = g_signs.to(R.device, non_blocking=True)  # almost never need to copy

    for i in range(3):
        for j in range(3):
            minors[:, :, i, j, :, :] = R[:, :, idx_i[i, j], idx_j[i, j]]

    minors = minors[:, :, :, :, 0, 0] * minors[:, :, :, :, 1, 1] - minors[:, :, :, :, 0, 1] * minors[:, :, :, :, 1, 0]
    cofactors = minors * signs[None, None]  # 3,3 -> B,N,3,3
    cofactors_t = cofactors.transpose(-2, -1)  # B, N, 3, 3
    determinant = R[:, :, 0, 0] * minors[:, :, 0, 0] - R[:, :, 0, 1] * minors[:, :, 0, 1] + R[:, :, 0, 2] * minors[:, :, 0, 2]  # B, N
    inverse = cofactors_t / (determinant[:, :, None, None] + eps)

    return inverse

# @torch.jit.script


def torch_inverse_3x3(R: torch.Tensor, EPS=1e-8):
    # n_batch, n_bones, 3, 3
    """
    a, b, c | m00, m01, m02
    d, e, f | m10, m11, m12
    g, h, i | m20, m21, m22
    """

    # convenient access
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    M = torch.empty_like(R)

    # determinant of matrix minors
    # fmt: off
    M[..., 0, 0] =   r11 * r22 - r21 * r12
    M[..., 1, 0] = - r10 * r22 + r20 * r12
    M[..., 2, 0] =   r10 * r21 - r20 * r11
    M[..., 0, 1] = - r01 * r22 + r21 * r02
    M[..., 1, 1] =   r00 * r22 - r20 * r02
    M[..., 2, 1] = - r00 * r21 + r20 * r01
    M[..., 0, 2] =   r01 * r12 - r11 * r02
    M[..., 1, 2] = - r00 * r12 + r10 * r02
    M[..., 2, 2] =   r00 * r11 - r10 * r01
    # fmt: on

    # determinant of matrix
    D = r00 * M[..., 0, 0] + r01 * M[..., 1, 0] + r02 * M[..., 2, 0]

    # inverse of 3x3 matrix
    M = M / (D[..., None, None] + EPS)

    return M


# @torch.jit.script
def torch_inverse_2x2(A: torch.Tensor, eps=torch.finfo(torch.float).eps):
    B = torch.zeros_like(A)
    # for readability
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]
    # slightly slower but save 20% of space (??) by
    # storing determinat inplace
    det = B[..., 1, 1]
    det = (a * d - b * c)
    det = det + eps
    B[..., 0, 0] = d / det
    B[..., 0, 1] = -b / det
    B[..., 1, 0] = -c / det
    B[..., 1, 1] = a / det
    return B


def mat2rt(A: torch.Tensor) -> torch.Tensor:
    """calculate 6D rt representation of blend weights and bones
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """

    # bw
    # 1. get blended transformation from bw and bones
    # 2. get quaternion from matrix
    # 3. get axis-angle from quaternion
    # 4. slice out the translation
    # 5. concatenation
    # A = blend_transform(input, batch.A)

    r = transforms.quaternion_to_axis_angle(transforms.matrix_to_quaternion(A[..., :3, :3]))  # n_batch, n_points, 3
    t = A[..., :3, 3]  # n_batch, n_points, 3, drops last dimension
    rt = torch.cat([r, t], dim=-1)
    return rt


def screw2rt(screw: torch.Tensor) -> torch.Tensor:
    return mat2rt(transforms.se3_exp_map(screw.view(-1, screw.shape[-1]))).view(*screw.shape)


def blend_transform(bw: torch.Tensor, A: torch.Tensor):
    """blend the transformation
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A = (bw.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(-4)).sum(dim=-3)
    return A


def tpose_points_to_ndc_points(pts: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def tpose_dirs_to_ndc_dirs(dirs: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    R = M[:, :3, :3]
    dirs = dirs.matmul(R.mT)
    return dirs


def world_dirs_to_pose_dirs(wdirs, R):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    """
    pts = torch.matmul(wdirs, R)
    return pts


def pose_dirs_to_world_dirs(pdirs, R):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    """
    pts = torch.matmul(pdirs, R.mT)
    return pts


def world_points_to_pose_points(wpts, R, Th):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    if Th.ndim == 2:
        Th = Th[..., None, :] # add fake point dimension
    pts = torch.matmul(wpts - Th, R)
    return pts


def pose_points_to_world_points(ppts, R, Th):
    """
    ppts: n_batch, n_points, 3
    R: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    if Th.ndim == 2:
        Th = Th[..., None, :] # add fake point dimension
    pts = torch.matmul(ppts, R.mT) + Th
    return pts


def pose_dirs_to_tpose_dirs(ddirs, bw=None, A=None, A_bw=None, R_inv=None):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    # since the blended rotation matrix is not a pure rotation anymore, we transform with inverse transpose
    R = A_bw[..., :3, :3]  # never None
    R_transpose = R.mT  # inverse transpose of inverse(R)
    pts = torch.sum(R_transpose * ddirs.unsqueeze(-2), dim=-1)
    return pts


def pose_points_to_tpose_points(ppts: torch.Tensor, bw=None, A=None, A_bw=None, R_inv=None):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    pts = ppts - A_bw[..., :3, 3]
    R_inv = torch_inverse_3x3(A_bw[..., :3, :3]) if R_inv is None else R_inv
    pts = torch.sum(R_inv * pts.unsqueeze(-2), dim=-1)
    return pts


def tpose_points_to_pose_points(pts, bw=None, A=None, A_bw=None, R_inv=None):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    R = A_bw[..., :3, :3]
    pts = torch.sum(R * pts.unsqueeze(-2), dim=-1)
    pts = pts + A_bw[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw=None, A=None, A_bw=None, R_inv=None):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw

    # since the blended rotation matrix is not a pure rotation anymore, we transform with inverse transpose
    R_inv = torch_inverse_3x3(A_bw[..., :3, :3]) if R_inv is None else R_inv
    R_inv_trans = R_inv.mT  # inverse transpose of the rotation

    pts = torch.sum(R_inv_trans * ddirs.unsqueeze(-2), dim=-1)
    return pts


world_points_to_view_points = world_points_to_pose_points  # input w2c, apply w2c
view_points_to_world_points = pose_points_to_world_points  # input w2c, inversely apply w2c


def grid_sample_blend_weights(grid_coords, bw):
    # the blend weight is indexed by xyz
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]
    return bw


def pts_sample_blend_weights_surf(pts, verts, faces, values) -> torch.Tensor:
    # surf samp 126988 pts: 127.36531300470233
    # b, n, D
    bw, dists = sample_closest_points_on_surface(pts, verts, faces, values)
    bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
    return bw.permute(0, 2, 1)  # b, D+1, n


def pts_sample_blend_weights_vert(pts, verts, values) -> torch.Tensor:
    # b, n, D
    bw, dists = sample_closest_points(pts, verts, values)
    bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
    return bw.permute(0, 2, 1)  # b, D+1, n


def pts_sample_blend_weights_vert_blend(pts, verts, values, K=5) -> torch.Tensor:
    # vert samp K=5 126988 pts: 6.205926998518407
    # b, n, D
    bw, dists = sample_blend_K_closest_points(pts, verts, values, K)
    bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
    return bw.permute(0, 2, 1)  # b, D+1, n
# BLENDING


def pts_sample_blend_weights(pts, bw, bounds):
    """sample blend weights for points
    pts: n_batch, n_points, 3
    bw: n_batch, d, h, w, 25
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]

    return bw


def grid_sample_A_blend_weights(nf_grid_coords, bw):
    """
    nf_grid_coords: batch_size x n_samples x 24 x 3
    bw: batch_size x 24 x 64 x 64 x 64
    """
    bws = []
    for i in range(24):
        nf_grid_coords_ = nf_grid_coords[:, :, i]
        nf_grid_coords_ = nf_grid_coords_[:, None, None]
        bw_ = F.grid_sample(bw[:, i:i + 1],
                            nf_grid_coords_,
                            padding_mode='border',
                            align_corners=True)
        bw_ = bw_[:, :, 0, 0]
        bws.append(bw_)
    bw = torch.cat(bws, dim=1)
    return bw


def get_sampling_points(bounds, n_samples):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    x_vals = torch.rand([sh[0], n_samples])
    y_vals = torch.rand([sh[0], n_samples])
    z_vals = torch.rand([sh[0], n_samples])
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts


def forward_node_graph(verts: torch.Tensor, graph_rt: torch.Tensor, graph_nodes: torch.Tensor, graph_bones: torch.Tensor, graph_weights: torch.Tensor) -> torch.Tensor:
    n_batch = graph_rt.shape[0]
    verts = verts.expand(n_batch, *verts.shape[1:])
    graph_nodes = graph_nodes.expand(n_batch, *graph_nodes.shape[1:])
    graph_bones = graph_bones.expand(n_batch, *graph_bones.shape[1:])
    graph_weights = graph_weights.expand(n_batch, *graph_weights.shape[1:])

    # graph_bones: B, V, 4
    r, t = graph_rt.split([3, 3], dim=-1)
    R = batch_rodrigues(r.view(-1, 3)).view(n_batch, -1, 3, 3)
    vi = verts[..., None, :].expand(n_batch, -1, graph_bones.shape[-1], -1)  # B, V, 4, 3

    pj = graph_nodes[torch.arange(n_batch)[..., None, None], graph_bones]  # B, V, 4, 3
    tj = t[torch.arange(n_batch)[..., None, None], graph_bones]  # translation B, V, 4, 3
    Rj = R[torch.arange(n_batch)[..., None, None], graph_bones]  # rotation B, V, 4, 3, 3

    wj = graph_weights[..., None].expand(-1, -1, -1, 3)  # B, V, 4, 3
    vj = Rj.matmul((vi - pj)[..., None])[..., 0] + pj + tj  # B, V, 4, 3
    vi = (vj * wj).sum(dim=-2)
    return vi


def forward_deform_lbs(cverts: torch.Tensor, deform: torch.Tensor, weights: torch.Tensor, A: torch.Tensor, R: torch.Tensor = None, T: torch.Tensor = None, big_A=None) -> torch.Tensor:
    n_batch = A.shape[0]
    weights = weights.expand(n_batch, *weights.shape[1:])
    if deform is not None:
        tverts = cverts + deform
    else:
        tverts = cverts
    if big_A is not None:
        tverts = pose_points_to_tpose_points(tverts, weights, big_A)
    pverts = tpose_points_to_pose_points(tverts, weights, A)
    if R is not None and T is not None:
        wverts = pose_points_to_world_points(pverts, R, T)
    else:
        wverts = pverts
    return wverts


def inverse_deform_lbs(wverts: torch.Tensor, deform: torch.Tensor, weights: torch.Tensor, A: torch.Tensor, R: torch.Tensor, T: torch.Tensor, big_A=None) -> torch.Tensor:
    n_batch = deform.shape[0]
    weights = weights.expand(n_batch, *weights.shape[1:])
    pverts = world_points_to_pose_points(wverts, R, T)
    tverts = pose_points_to_tpose_points(pverts, weights, A)
    if big_A is not None:
        tverts = tpose_points_to_pose_points(tverts, weights, big_A)
    cverts = tverts - deform
    return cverts


def bilinear_interpolation(input: torch.Tensor, shape: List[int]) -> torch.Tensor:
    # input: B, H, W, C
    # shape: [target_height, target_width]
    return F.interpolate(input.permute(0, 3, 1, 2), shape, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)


def rand_sample_sum_to_one(dim, samples, device='cuda', negative_one=False):
    # negative_one: allow sampling to negative one?
    exp_sum = (0.5 * (dim - 1))
    bbweights = torch.rand(samples, dim - 1, device=device)  # 1024, 5
    bbweights_sum = bbweights.sum(dim=-1)
    extra_mask = bbweights_sum > exp_sum
    bbweights[extra_mask] = 1 - bbweights[extra_mask]
    last_row = (bbweights_sum - exp_sum).abs()
    bbweights = torch.cat([bbweights, last_row[..., None]], dim=-1)
    bbweights = bbweights / exp_sum

    if negative_one:
        bbweights = bbweights * 2 - 1 / dim
    return bbweights
    # bbweights = bbweights / (bbweights.sum(dim=-1, keepdim=True) + eps) # MARK: wrong normalization
    # __import__('ipdb').set_trace()


def linear_sample_sum_to_one(dim, samples, device='cuda', multiplier=5.0):
    interval = dim - 1
    samples_per_iter = samples // interval
    samples_last_iter = samples - (interval - 1) * samples_per_iter

    # except last dimension
    weights = torch.zeros(samples, dim, device=device)
    for i in range(interval - 1):
        active = torch.linspace(1, 0, samples_per_iter, device=device)
        active = active - 0.5
        active = active * multiplier
        active = active.sigmoid()
        active = active - 0.5
        active = active / active.max() / 2
        active = active + 0.5
        next = 1 - active
        weights[i * samples_per_iter:i * samples_per_iter + samples_per_iter, i] = active
        weights[i * samples_per_iter:i * samples_per_iter + samples_per_iter, i + 1] = next

    active = torch.linspace(1, 0, samples_last_iter, device=device)
    next = 1 - active
    weights[(interval - 1) * samples_per_iter:, interval - 1] = active
    weights[(interval - 1) * samples_per_iter:, interval] = next

    return weights


def interpolate_poses(poses, bbweights):
    Rs = axis_angle_to_matrix(poses)
    bbRs: torch.Tensor = torch.einsum('sn,nbij->sbij', bbweights, Rs)
    U, S, Vh = bbRs.svd()
    V = Vh.mH
    # __import__('ipdb').set_trace()
    bbRs = U.matmul(V)
    bbposes = quaternion_to_axis_angle(matrix_to_quaternion(bbRs))
    return bbposes


def interpolate_shapes(shapes, bbweights):
    # bbposes: torch.Tensor = torch.einsum('sn,nvd->svd', bbweights, poses)  # FIXME: curious shape shrinking
    bbshapes: torch.Tensor = torch.einsum('sn,nvd->svd', bbweights, shapes)  # FIXME: curious shape shrinking
    # bbdeformed: torch.Tensor = bbshapes + optim_tpose.verts[None]  # resd to shape
    return bbshapes
