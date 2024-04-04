import torch
import numpy as np
from torch import nn
from math import prod
from functools import partial
from typing import Callable, Union, Tuple, List

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.mesh_utils import moller_trumbore
from lib.utils.log_utils import print_colorful_stacktrace, log, run
from lib.utils.relight_utils import gen_light_xyz, Microfacet, linear2srgb, reflect, expand_envmap_probe, sample_envmap_image, add_light_probe, srgb2linear, expand_envmap_xyz
from lib.utils.net_utils import get_near_far_aabb, normalize, get_rays, compute_ground_tris, take_gradient, batch_aware_indexing, multi_gather, multi_scatter, sdf_to_occ, multi_scatter_, chunkify, volume_rendering, resize_image
from lib.utils.blend_utils import tpose_dirs_to_pose_dirs, pose_dirs_to_tpose_dirs, pose_points_to_tpose_points, tpose_points_to_pose_points
from lib.networks.relight.relight_network import Network
net_chunkify = chunkify(chunk_size=cfg.network_chunk_size, dim=-2, merge_dims=True)
pix_chunkify = chunkify(chunk_size=cfg.render_chunk_size, dim=-2, merge_dims=True)


@torch.no_grad()
@net_chunkify  # avoid large chunks of sphere tracing, might be too memory intensive
def sphere_tracing(ray_o: torch.Tensor,  # B, P, 3
                   ray_d: torch.Tensor,  # B, P, 3
                   near: torch.Tensor,  # B, P, 1 or B, P
                   far: torch.Tensor,  # B, P, 1 or B, P

                   sdf_decoder: Callable[[torch.Tensor], torch.Tensor],
                   world_to_can_transform: Callable[[torch.Tensor], torch.Tensor],  # will return A to transform from world to canonical
                   can_to_world_transform: Callable[[torch.Tensor], torch.Tensor],  # will return A to transform from canonical to world
                   iter: int = cfg.sphere_tracing.iter,
                   tan_i: Union[float, torch.Tensor] = cfg.sphere_tracing.tan_i,  # inverse of tan(theta)
                   relax: Union[float, torch.Tensor] = cfg.sphere_tracing.relax,
                   offset: Union[float, torch.Tensor] = cfg.sphere_tracing.offset,
                   eps: float = cfg.sphere_tracing.eps,
                   shadow_skip_iter: int = cfg.sphere_tracing.shadow_skip_iter,
                   clay_book: bool = not cfg.no_claybook,  # use banding removing mentioned in clay book
                   soft_shadow: bool = False,  # defaults to surface tracing
                   mode='hdq',  # choices = ['hdq', 'world', 'can', 'curve']
                   **kwargs,
                   ):
    """
    Vanilla Sphere Tracing alogrithm, augmented with my own implementation offset and relaxation trick.
    For surface intersection, we added a linear interpolation approximation for better surface points.
    For soft shadow effect, we used the claybook trick to remove banding artifacts.

    This is a fixed step sphere tracing algorithm, since we need to perform almost similar operation on all tensors.
    The number of iterations is pre-determined, and is where most of the computation is spent, and provides a controllable
    handle for performance-final results trade-off.
    We've tried an early termination implementation, by masking valid or invalid pixels after every tracing iteration.
    It did not go well, almost as slow as the fixed step implementation with half the number of computations, worse convergence.

    Fixing the number of iterations to small number -> typically 16, means incorrect distance values (maybe produced by naive
    world space sphere tracing, will be propagated to the next iteration, and the final result will be incorrect after the designated
    number of iterations are finished.

    This tracing utilize the fact that the distance values are signed, instead of unsigned.
    When we're inside the actual surface, the query algorithm would produce a good negative number for us to perform backward tracing.
    Along the ray, when outside, we go forward, when inside, we go backward thus we can safely ignore the convergence condition of the
    original Sphere Tracing algorithm for better surface intersection computation.

    For Neural SDF, network evaluations are very-very-very heavy, we'd like to avoid them as much as possible, thus smaller number of iterations.
    If we can somehow increase the quality of the distance function, we can reduce the number of iterations, and thus increase performance.

    When the near distance is far from surface, the first iteration would actually remove almost all the distances.
    This is actually faster than performing fancy rasterization and find a closer distance.

    Curve Tracing is the non-linear version of the Sphere Tracing algorithm
    Since previous neural avatars define distance in canonical space, the tracing is limited to canonical space \cite{Non-linear sphere tracing for rendering deformed signed distance fields}.
    https://cs.dartmouth.edu/~wjarosz/publications/seyb19nonlinear.html

    This function is to show that the ray in world space would produce too complex a curve in canonical space during inverse LBS, leading to incorrect sphere tracing results.

    Ideally, we would generate an image with curve tracing and our deformed sphere tracing, and compare them with GT to show curve tracing produces invalid results.

    Procedure for curved tracing:
    1. Warp pts, dir from world to canonical
    2. Query the canonical distance value
    3. Perform tracing in canonical, getting next pts
    4. Warp pts back to world space from its new canonical position
    5. Warp pts and dir to canonial from its new world position
    6. Repeat step 2 through 5 until convergence

    early ternimation implementation does not seem to work so well
    when computing soft shadow, only occ matters, other can be arbitrary
    when not computing soft shadow, we need the surface intersection to be accurate

    sdf_decoder should just advance the ray, and return the updated world space direction and ray_o
    should we still consider soft shadow
    think about how we should describe the curve tracing process?
    1. ray_o is defined in canonical space
    2. ray_d is defined by the jacobian of the direction in canonical space
    2. d1 is computed in canonical space

    We have three variants of Sphere Tracing to ablate:
    1. HDQ World (Ours): with hierarchical distance query, directly perform world space sphere tracing of HDQ's distance
    2. Naive World: since distances are defined in canonical, only update sphere tracing w.r.t. canonical distance -> no convergence in iter
    3. Naive Canonical: since distances are defined in canonical, warp ray_o and ray_d to canonical, tracing, warp back -> incorrect intersection
    4. Curved Canonical: since distances are defined in canonical, warp ray_o and ray_d to canonical, trace using warped ray_d in every iteration -> incorrect intersection

    For now, the Sphere Tracing is only ablated on surface intersection, should we consider soft-visibility? No.
    """

    if isinstance(near, torch.Tensor) and near.ndim < ray_o.ndim:
        near = near[..., None]
    if isinstance(far, torch.Tensor) and far.ndim < ray_o.ndim:
        far = far[..., None]
    if not soft_shadow:
        tan_i = cfg.sphere_tracing.tan_i  # hard shadow
    else:
        tan_i = cfg.sphere_tracing.tan_i_multiplier * tan_i  # a little bit harder shadow

    ones = torch.ones(*ray_o.shape[:-1], 1, device=ray_o.device)
    near = ones * near  # in case they are scalars
    far = ones * far
    tan = ones / tan_i
    off = ones * offset
    rlx = ones * relax
    occ = ones  # minimum accumulated occlusion
    d0 = ones * 1e9  # r_i-1
    d1 = ones * 1e9  # r_i
    cd = ones * 1e9  # closest distance
    dt = ones * 1e9  # d_i, the step size for every iteration
    st = far  # closest t (not always updated)
    ot = far  # occlusion's t (may contain d1 < 0), might cause nan when computing gradients on occ values in network
    t = near

    def apply_transform(ray_f: torch.Tensor, ray_d: torch.Tensor, A: torch.Tensor):
        if ray_d is not None:
            return tpose_points_to_pose_points(ray_f, A_bw=A), tpose_dirs_to_pose_dirs(ray_d, A_bw=A)
        else:
            return tpose_points_to_pose_points(ray_f, A_bw=A)

    def world_to_can(ray_f: torch.Tensor, ray_d: torch.Tensor = None):
        A = world_to_can_transform(ray_f)
        return apply_transform(ray_f, ray_d, A)

    def can_to_world(ray_f: torch.Tensor, ray_d: torch.Tensor = None):
        A = can_to_world_transform(ray_f)
        return apply_transform(ray_f, ray_d, A)

    if mode == 'can':
        ray_o, ray_d = world_to_can(ray_o, ray_d)

    for i in range(iter):
        ray_f = ray_o + t * ray_d  # this is the front of the ray during tracing

        # perform sdf query (heavy)
        if mode == 'world':  # the decoder is a canonical space decoder
            d1 = sdf_decoder(world_to_can(ray_f), **kwargs)  # canonical distance, w2c transformation
        else:  # the decoder and ray front are defined in the same space
            d1 = sdf_decoder(ray_f, **kwargs)

        # this will cause strange artifact when tracing for self-occlusion
        # but it removes a lot of banding for ground visibility map
        # might be some faulty implementation issue? possible to look through the source?
        # soft shadow stuff, should happen before off and relax correction
        if soft_shadow and clay_book and i >= shadow_skip_iter:
            dx0 = d0 + rlx * d0 + off  # repeated computation here
            dx1 = d1 + rlx * d1 + off  # find changed distance
            dy = (dx1 ** 2) / (2 * dx0)  # find line between intersections
            dx = ((dx1 ** 2 - dy ** 2).sqrt() - off) / (1 + rlx)  # compute actual closest distance
            cls = dx.clip(0) / (t - dy).clip(near).clip(eps) / (tan * 2)  # darkest penumbra (closest surface point)
            msk = (cls < occ) & (i >= shadow_skip_iter)  # main mask
            msk = msk & (dy < t)  # make sure ot is valid
            msk = msk & (dx1 > 0)  # make sure no negative number
            msk = msk & (dx0 > 0)  # make sure no negative number
            msk = msk & (dx > 0)  # make sure no negative number
            msk = msk & (dy > 0)  # make sure no negative number
            msk = msk & (dy < dx0)  # should not be too long
            # nan in binary op -> False
            ot = torch.where(msk, t - dy, ot)
            occ = torch.where(msk, cls, occ)

        # find soft shadow coefficient
        if i >= shadow_skip_iter:
            cls = d1.clip(0) / t.clip(near).clip(eps) / (tan * 2)
            msk = (cls < occ) & (i >= shadow_skip_iter)
            ot = torch.where(msk, t, ot)
            occ = torch.where(msk, cls, occ)

        # there's a closer intersection point (cloest to surface)
        if not soft_shadow:
            d1_udf = d1.abs()
            d0_udf = d0.abs()

        # there's a better intersection point (linearly interpolated) (an actual ray hit)
        if not soft_shadow:
            msk = d0.sign() != d1.sign()  # consider entry and exit
            st = torch.where(msk, (t - dt * (d1_udf / (d0_udf + d1_udf + eps)).clip(0, 1)), st)
            off = torch.where(msk, 0, off)  # let it converge normally
            rlx = torch.where(msk, 0, rlx)  # let it converge normally

        # make the surface better around actual edge
        if not soft_shadow:
            msk = d1_udf < cd
            cd = torch.where(msk, d1_udf, cd)
            st = torch.where(msk, t, st)

        # relax to get the next value
        dt = d1 + rlx * d1 + off
        t = t + dt

        # remove far away cases
        t = torch.minimum(t, far)  # will remove strange nan values for now
        t = torch.maximum(t, near)

        # control variable
        d0 = d1

    surf = ray_o + st * ray_d  # surface intersection point
    edge = ray_o + ot * ray_d  # surface intersection point
    if mode == 'can':
        surf = can_to_world(surf)
        edge = can_to_world(edge)

    return surf, edge, occ, st, ot  # B, P, 3; B, P


@torch.no_grad()
@net_chunkify  # avoid large chunks of sphere tracing, might be too memory intensive
def softer_shadow(ray_o: torch.Tensor,  # B, P, 3
                  ray_d: torch.Tensor,  # B, P, 3
                  near: torch.Tensor,  # B, P, 1 or B, P
                  far: torch.Tensor,  # B, P, 1 or B, P

                  sdf_decoder: Callable[[torch.Tensor], torch.Tensor],
                  iter: int = cfg.sphere_tracing.iter,
                  tan_i: Union[float, torch.Tensor] = cfg.sphere_tracing.tan_i,  # inverse of tan(theta)
                  eps: float = cfg.sphere_tracing.eps,

                  *args,
                  **kwargs,
                  ):
    # convert illegal inputs
    near = near[..., None] if isinstance(near, torch.Tensor) and near.ndim < ray_o.ndim else near
    far = far[..., None] if isinstance(far, torch.Tensor) and far.ndim < ray_o.ndim else far

    # assume inputs are all tensors
    ones = torch.ones(*ray_o.shape[:-1], 1, device=ray_o.device)
    near = ones * near  # in case they are scalars
    far = ones * far
    tan = ones / tan_i
    occ = ones  # minimum accumulated occlusion
    t = near

    for i in range(iter):
        # the heavy lifting happens here
        h = sdf_decoder(ray_o + t * ray_d, **kwargs) + t * tan  # the magic happens here
        occ = torch.minimum(occ, h.clip(eps) / t.clip(eps) / (2 * tan))
        # t = t + h - t * tan
        t = t + h * torch.rsqrt(t + 1)  # stepping with inverse square root -> TODO: larger number of iterations?
        # t = t + h  # stepping

        t = torch.maximum(t, near)  # sanity checks
        t = torch.minimum(t, far)

    # return surf, edge, occ, st, ot  # B, P, 3; B, P
    edge = ray_o + t * ray_d
    surf = edge
    ot = t
    st = ot
    return surf, edge, occ, st, ot


@torch.no_grad()  # sphere tracing should not expect gradients and light visibility is essentially sphere tracing
def light_visibility(
    surf: torch.Tensor,  # B, P, 3 (surface points)
    norm: torch.Tensor,  # B, P, 3
    acc: torch.Tensor,  # B, P
    xyz: torch.Tensor,  # eH, eW, 3
    sharp: torch.Tensor,  # eH, eW (computed from area of the light source, tan(theta), called sharpness)
    sphere_tracing_decoder: Callable[[torch.Tensor], torch.Tensor],
    bbox: Union[None, torch.Tensor] = None,  # B, 2, 3
    near_offset: float = cfg.sphere_tracing.near_offset,
    soft_shadow: bool = not cfg.no_dfss,
    far_offset: float = cfg.env_r,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # prepare shapes
    B, P, _ = surf.shape
    eH, eW, _ = xyz.shape
    F = eH * eW * P

    ray_o = surf[:, None, None].expand(B, eH, eW, P, 3).reshape(B, F, 3)  # B, P, 3 -> B, H, W, P, 3, ray_t termination is the surface point
    ray_d = normalize(xyz)[None, :, :, None, :].expand(B, eH, eW, P, 3).reshape(B, F, 3)
    # ray_d = normalize(ray_o - bbox.mean(dim=-2, keepdim=True) + (xyz / 10)[None, :, :, None].expand(B, eH, eW, P, -1).reshape(B, F, -1))
    tan_i = sharp[None, :, :, None, None].expand(B, eH, eW, P, 1).reshape(B, F, 1)  # H, W -> B, H, W, P, 1
    norm = norm[:, None, None].expand(B, eH, eW, P, 3).reshape(B, F, 3)
    acc = acc[:, None, None].expand(B, eH, eW, P).reshape(B, F)

    # compute dot product (cosine of normal and light direction)
    ldot = (ray_d * norm).sum(dim=-1)  # B, H, W, P, this is l_dot_n and this should be omitted

    # early termination for ablation study
    if cfg.no_visibility:
        lvis = torch.ones_like(ldot)
        return lvis.view(B, eH, eW, P), ldot.view(B, eH, eW, P)  # this is stupid
    if cfg.local_visibility:
        lvis = (ldot > 0).float()
        return lvis.view(B, eH, eW, P), ldot.view(B, eH, eW, P)

    # reduce number of points to perform sphere tracing on
    lfrt_mask = (ldot > 0) & (acc > 0)
    _, lfrt, _ = batch_aware_indexing(lfrt_mask, ldot * acc)  # MARK: SYNC

    near = torch.ones_like(ray_d[..., -1:]) * near_offset
    far = torch.ones_like(ray_d[..., -1:]) * far_offset
    # render the bounding box out, and filter redundant points for faster computation
    if bbox is not None:
        near_lfrt, far_lfrt = get_near_far_aabb(bbox, multi_gather(ray_o, lfrt), multi_gather(ray_d, lfrt), return_raw=True)
        near_lfrt, far_lfrt = near_lfrt[..., None].clip(near_offset), far_lfrt[..., None].clip(near_offset)
        near = multi_scatter(near, lfrt, near_lfrt)
        far = multi_scatter(far, lfrt, far_lfrt)
        lbox_mask = multi_scatter(torch.zeros_like(lfrt_mask), lfrt, (near_lfrt < far_lfrt)[..., 0], dim=-1)
        lren_mask = lfrt_mask & lbox_mask  # integrate mask at box, a little bit wasted computation
        _, lren, _ = batch_aware_indexing(lren_mask, ldot * acc * lbox_mask)  # MARK: SYNC
    else:
        lren_mask = lfrt_mask
        lbox_mask = torch.ones_like(lfrt_mask)
        lren = lfrt

    # assume points are always on the outside
    # will never produce hard enough self-shadows? -> might always just be closer to the previous surface -> can we just normal to test that out?
    ray_o = multi_gather(ray_o, lren)
    ray_d = multi_gather(ray_d, lren)
    near = multi_gather(near, lren)
    far = multi_gather(far, lren)
    tan_i = multi_gather(tan_i, lren)
    # TODO: bug in chunkify with merge_dims
    # TODO: black points on all directions when tracing soft shadow
    surf, edge, occ, st, ot = sphere_tracing_decoder(
        ray_o,
        ray_d,
        near,
        far,
        tan_i=tan_i,
        soft_shadow=soft_shadow,
        **kwargs,
    )
    lvis = multi_scatter(ray_o.new_zeros(*acc.shape), lren, occ[..., 0], dim=-1)
    lvis = lvis * lbox_mask + 1 * ~lbox_mask  # inside box but unable to see the light: ~lbox & ~lfrt
    lvis = lvis * lfrt_mask + 0 * ~lfrt_mask

    return lvis.view(B, eH, eW, P), ldot.view(B, eH, eW, P)


def evaluate_brdf(
    surf: torch.Tensor,  # B, P, 3 # surface points
    albedo: torch.Tensor,  # B, P, 1
    rough: torch.Tensor,  # B, P, 1
    norm: torch.Tensor,  # B, P, 3
    cam: torch.Tensor,  # B, P, 3, camera position
    xyz: torch.Tensor,  # eH, eW, 3, light position
    microfacet: Microfacet,  # a microfacet pbr material
) -> torch.Tensor:  # returns linear rgb

    B, P, C = surf.shape
    eH, eW, C = xyz.shape
    if B != eH and P != eW:  # expanding for multiple light sources, # ! could be dangerous: ndim == ndim
        surf2light = normalize(xyz[None, :, :, None] - surf[:, None, None])  # B, eH, eW, P, 3
    else:
        surf2light = normalize(xyz - surf)  # B, P, 3
    surf2cam = normalize(cam - surf)  # B, P, 3
    brdf = microfacet(surf2light, surf2cam, norm, albedo, rough)  # B, eH, eW, P, 3

    return brdf


def evaluate_shade(
    lvis: torch.Tensor,  # B, eH, eW, P
    ldot: torch.Tensor,  # B, eH, eW, P
    area: torch.Tensor,  # eH, eW,
    light: torch.Tensor,  # B, eH, eW, P, 3
):
    shade = lvis[..., None] * ldot[..., None] * area[None, :, :, None, None] * light  # memory?
    return shade


blend_keys = ['rgb_map',
              'rfl_map',
              'surf_map',
              'albedo_map',
              'roughness_map',
              'norm_map',
              'cpts_map',
              'bpts_map',
              'spec_map',
              'depth_map',
              'lvis_map',  # MARK: MEM
              'ldot_map',  # MARK: MEM
              'brdf_map',  # MARK: MEM
              'shade_map',  # MARK: MEM
              ]


def alpha_blend(acc: torch.Tensor, inds: torch.Tensor, target: torch.Tensor, value: torch.Tensor):
    # value could be light-wise or already merged
    # light-wise: target: B, eH, eW, F, C (or maybe C does not exist)
    # already merged: target: B, P, C
    ndim = target.ndim
    if ndim == 4 or ndim == 2:
        target = target[..., None]
        value = value[..., None]
    if ndim >= 4:
        inds = inds[:, None, None, ..., None]
        acc = acc[:, None, None, ..., None]
        scatter = value.new_zeros(*target.shape[:4], *value.shape[4:])
    else:
        inds = inds[..., None]
        acc = acc[..., None]
        scatter = value.new_zeros(*target.shape[:2], *value.shape[2:])
    scatter = multi_scatter_(scatter, inds, value)
    merged = target * acc + scatter * (1 - acc)  # fill in the background with 1
    if ndim == 4 or ndim == 2:
        merged = merged[..., 0]
    return merged


def alpha_times(acc: torch.Tensor, value: torch.Tensor):
    ndim = value.ndim
    if ndim == 4 or ndim == 2:
        value = value[..., None]
    if ndim >= 4:
        acc = acc[:, None, None, ..., None]
    else:
        acc = acc[..., None]
    merged = value * acc  # fill in the background with 1
    if ndim == 4 or ndim == 2:
        merged = merged[..., 0]
    return merged


def blend_output_(acc: torch.Tensor, inds: torch.Tensor, grd: dotdict, ret: dotdict):
    # note that this functions modifies ret: dotdict
    # this will only blend things together for visualization
    # when performing rendering, we still need to separate the two passes
    # otherwise the edges will look janky
    # (a * alpha + b * (1 - alpha)) * (c * alpha + d * (1 - alpha)) !=
    # a * c * alpha + b * d * (1 - alpha)

    # blend output values together
    for key in blend_keys:
        if key in ret and key in grd:
            ret[key] = alpha_blend(acc, inds, grd[key], ret[key])  # save some memory
        if key not in ret and key in grd:
            ret[key] = alpha_times(acc, grd[key])
        # if key in ret and key not in grd:
        #     ret[key] = alpha_times(acc, ret[key])
    ret.acc_map = alpha_blend(acc, inds, torch.zeros_like(acc), ret.acc_map)  # save some memory

    return ret


def alpha_output_(acc: torch.Tensor, ret: dotdict):
    # blend output values together
    for key in blend_keys:
        if key in ret:
            ret[key] = alpha_times(acc, ret[key])  # save some memory

    return ret


def render_ground(
    ray_o: torch.Tensor,  # B, P, 3 the rays to render
    ray_d: torch.Tensor,  # B, P, 3 the rays to render
    acc: torch.Tensor,  # B, P, 3 the acc of the ray to render
    xyz: torch.Tensor,  # eH, eW, 3
    area: torch.Tensor,  # eH, eW
    sharp: torch.Tensor,  # eH, eW
    bbox: torch.Tensor,  # B, 2, 3, human bounding box
    envmap: dotdict,  # envmap.probe is for rendering with the light probe, envmap.image is the 8k environment map

    # network queries for hdq & raw material output
    shadow_tracing_decoder: Callable[[torch.Tensor], torch.Tensor],
):

    # generate mask to alpha blend the foreground and background: grp_map
    B, P, _ = ray_o.shape
    eH, eW, _ = xyz.shape

    norm = ray_o.new_tensor(cfg.ground_normal)
    orig = ray_o.new_tensor(cfg.ground_origin)
    norm = normalize(norm)
    tris = compute_ground_tris(orig, norm)  # 3, 3
    t = moller_trumbore(ray_o.view(B * P, 3), ray_d.view(B * P, 3), tris.view(1, 3, 3))[-1].view(B, P, 1)  # B, P, 1
    surf = ray_o + t * ray_d  # B, P, 3
    norm = norm[None, None].expand(B, P, 3)  # B, P, 3

    lvis, ldot = light_visibility(surf, norm, acc, xyz, sharp, shadow_tracing_decoder, bbox, **cfg.env_lvis)

    # prepare ground albedo and roughness
    # use provided environment map as background, need to address the normal cutoff issue
    # maybe we only consider light visibility? no shading? will not look good?
    if cfg.ground_attach_envmap:
        if 'image' in envmap:
            albedo = sample_envmap_image(envmap.image, ray_d)  # B, P, 3
        else:
            albedo = sample_envmap_image(envmap.probe, ray_d)  # B, P, 3
    else:
        albedo = torch.ones_like(surf) * lvis.new_tensor(cfg.ground_albedo)  # B, P, 3

    # modify the ground shading results to ease into the actual environment map's color
    # we start the blending from distance 0.5 * cfg.env_r through 0.6 * cfg.env_r
    dist = torch.where(t[..., 0] <= 0, 1e9, (surf - orig).norm(dim=-1))[:, None, None].expand(ldot.shape)  # when viewing upward, no ground please
    weight = ((dist - cfg.env_r) / cfg.env_r).clip(0, 1)  # blending weights

    # perform the actual blending
    ldot = (normalize(xyz[None, :, :, None]) * norm[:, None, None]).sum(dim=-1)
    lvis = lvis * (1 - weight) + torch.ones_like(lvis.mean(dim=-1, keepdim=True)) * weight  # only consider shadow on the ground
    brdf = albedo[:, None, None] / np.pi  # no material for ground shading... per the Microfacet model's implementation

    # perform the actual shading with lvis, ldot and given light
    # light = sample_envmap_image(envmap.probe, xyz - surf)  # B, eH, eW, P, 3
    surf2light = normalize(xyz[:, :, None] - torch.zeros_like(surf[:, None, None]))
    light = sample_envmap_image(envmap.probe, surf2light)  # B, eH, eW, P, 3
    if cfg.only_visibility:
        # debugging option
        ldot = torch.ones_like(ldot)  # make the final image look uniform
        light = light.mean(dim=-1, keepdim=True)  # single channel light for visibility
    shade = evaluate_shade(lvis, ldot, area, light)  # memory?
    rgb = brdf * shade
    rgb = rgb.sum(dim=1).sum(dim=1)
    if cfg.tonemapping_rendering:
        rgb = linear2srgb(rgb)
    # http://www.joshbarczak.com/blog/?p=272
    shade = shade.sum(1).sum(1) * cfg.shading_albedo / np.pi  # brdf should integrate to 1 on cosien solid angle

    # return values & update batch for some hacky side-effects
    ret = dotdict()
    ret.rgb_map = rgb  # should only blend final values
    ret.surf_map = surf
    ret.albedo_map = albedo
    ret.roughness_map = torch.ones_like(albedo[..., 0])
    ret.spec_map = shade / 20  # no specularity, use shading instead
    ret.norm_map = norm
    ret.shade_map = shade
    if cfg.vis_lvis_map: ret.shade_map = lvis.mean(1).mean(1)[..., None].expand(-1, -1, 3)
    if cfg.vis_ldot_map: ret.shade_map = ldot.mean(1).mean(1)[..., None].expand(-1, -1, 3)
    ret.shade_map = ret.shade_map * cfg.ground_shading_multiplier

    ret.cpts_map = torch.zeros_like(surf)
    ret.bpts_map = torch.zeros_like(surf)
    ret.depth_map = t[..., 0].clip(-cfg.env_r, cfg.env_r)  # depth
    if cfg.vis_novel_light:
        ret.lvis_map = lvis.view(B, prod(lvis.shape[1:3]), lvis.shape[-1]).permute(0, 2, 1)  # MARK: MEM, DIM
        ret.ldot_map = ldot.view(B, prod(ldot.shape[1:3]), ldot.shape[-1]).permute(0, 2, 1)  # MARK: MEM, DIM

    return ret


def render_human(ray_o: torch.Tensor,
                 ray_d: torch.Tensor,
                 near: torch.Tensor,
                 far: torch.Tensor,
                 envmap: dotdict,
                 xyz: torch.Tensor,
                 area: torch.Tensor,
                 sharp: torch.Tensor,
                 training: bool,
                 microfacet: Microfacet,

                 bbox: torch.Tensor,  # B, 2, 3, human bounding box
                 # network queries for hdq & raw material output
                 surface_tracing_decoder: Callable[[torch.Tensor], torch.Tensor],
                 shadow_tracing_decoder: Callable[[torch.Tensor], torch.Tensor],
                 sdf_decoder: Callable[[torch.Tensor], torch.Tensor],
                 net_decoder: Callable[[torch.Tensor], torch.Tensor],
                 ):

    # Surface tracing decoder with unsmooth distance values results in black points when smplh and the surface are not aligned
    surf, edge, occ, st, ot = surface_tracing_decoder(ray_o, ray_d, near, far)  # different implementation
    # surf, edge, occ, st, ot = shadow_tracing_decoder(ray_o, ray_d, near, far)

    depth_map = (surf[..., 0] - ray_o[..., 0]) / ray_d[..., 0]  # MARK: CHANNEL
    acc = 1 - occ[..., 0]  # MARK: CHANNEL & GRADIENT

    if cfg.check_bound_sdf:
        from easyvolcap.utils.color_utils import colormap
        from easyvolcap.utils.console_utils import log, run
        d_surf = sdf_decoder(surf)
        d_edge = sdf_decoder(edge)  # find the bleneded sdf value at ray termination point
        d = torch.where(acc[..., None] > 0, d_surf, d_edge)

        ret = dotdict()
        ret.acc_map = torch.ones_like(acc)  # hopefully?
        ret.rgb_map = colormap(d.abs() * 2)  # hopefully?
        return ret  # early stop for visualization purpose

    # filtered forward pass
    _, inds, _ = batch_aware_indexing(acc > 0, acc)  # MARK: SYNC

    # compute occ based on closest points (maybe inside surface but it doesn't matter)
    if training:  # only for adding gradients to geometry
        d = sdf_decoder(multi_gather(edge, inds))
        acc = 1 - d.clip(0) / multi_gather(ot, inds).clip(multi_gather(near[..., None], inds)).clip(cfg.sphere_tracing.eps) / (1 / cfg.sphere_tracing.tan_i * 2)
        acc = acc.clip(0, 1)
        # acc = multi_scatter_(ray_o.new_zeros(*ray_o.shape[:-1], occ.shape[-1]), inds, acc)
        acc = acc[..., 0]  # MARK: CHANNEL
    else:
        acc = multi_gather(acc, inds, dim=-1)  # MARK: CHANNEL

    # compute attributes on surface points
    surf = multi_gather(surf, inds)
    view = multi_gather(ray_d, inds)
    ray_o = multi_gather(ray_o, inds)
    depth_map = multi_gather(depth_map, inds, dim=-1)  # MARK: CHANNEL
    if cfg.n_samples == 1:
        zval = torch.full((1,), 0.5, device=surf.device, dtype=surf.dtype)
    else:
        zval = torch.linspace(0., 1., steps=cfg.n_samples, device=surf.device, dtype=surf.dtype)
    net_zval = zval * (2 * cfg.surf_sample_range) - cfg.surf_sample_range
    net_view = view[..., None, :].expand(-1, -1, cfg.n_samples, -1)
    net_surf = surf[..., None, :].expand(-1, -1, cfg.n_samples, -1) + net_zval[None, None, :, None] * net_view  # B, P, S, 3
    B, P, S, C = net_surf.shape
    ret = net_decoder(net_surf.reshape(B, P * S, C), net_view.reshape(B, P * S, C))
    raw = ret.raw.view(B, P, S, ret.raw.shape[-1])
    raw, occ = raw[..., :-1], raw[..., -1:]  # MARK: CHANNEL
    _, raw, occ = volume_rendering(raw, occ[..., -1], bg_brightness=cfg.bg_brightness)  # B, P, -1
    raw = raw / (occ[..., None] + 1e-8)  # invert values back after volume rendering (sum to one)
    raw = torch.cat([raw, occ[..., None]], dim=-1)  # B, P, -1

    # for computing loss?
    ret.acc_map = acc  # now they should be the same, should this be used in computing loss?
    if not training:
        ret.ray_o = ray_o
        ret.surf_map = surf
        ret.depth_map = depth_map

    if raw.shape[-1] == 3 + 1 + 3 + 1:  # is relighting (will not return rgb, needs rendering)
        # expand network output
        albedo, roughness, norm, occ = raw.split([3, 1, 3, 1], dim=-1)  # B, P, X
    elif raw.shape[-1] == 3 + 3 + 3 + 3 + 1 + 3 + 1:
        cpts, bpts, resd, albedo, roughness, norm, occ = raw.split([3, 3, 3, 3, 1, 3, 1], dim=-1)  # B, P, X
    elif raw.shape[-1] == 3 + 3 + 3 + 3 + 3 + 1:
        cpts, bpts, resd, norm, rgb, occ = raw.split([3, 3, 3, 3, 3, 1], dim=-1)
    elif raw.shape[-1] == 3 + 3 + 1:
        norm, rgb, occ = raw.split([3, 3, 1], dim=-1)
    else:
        raise NotImplementedError(f'Supported ret.raw shape: {ret.raw.shape}')

    # correct the faulty volume rendering process
    norm = torch.where(norm.sum(dim=-1, keepdim=True) == 0, torch.ones_like(norm), norm)  # remove zero values
    norm = normalize(norm)  # volume rendering is not good for normals

    if 'albedo' in locals():
        albedo = albedo.clip(cfg.albedo_bias, cfg.albedo_bias + cfg.albedo_slope)
        ret.volume_albedo = albedo
    if 'roughness' in locals():
        roughness = roughness.clip(cfg.roughness_bias, cfg.roughness_bias + cfg.roughness_slope)
        ret.volume_roughness = roughness

    # for better visualization
    if not training and cfg.albedo_multiplier > 0:
        if 'albedo' in locals():
            albedo = albedo * cfg.albedo_multiplier

    # for visualization
    if not training and cfg.rgb_as_albedo:
        if 'albedo' in locals():
            albedo = srgb2linear(rgb)

    expanding_keys = ['lvis_map',
                      'ldot_map',
                      'shade_map',
                      'spec_map',
                      'rgb_map',
                      'cpts_map',
                      'bpts_map',
                      'resd_map',
                      'norm_map',
                      'albedo_map',
                      'roughness_map',
                      'surf_map',
                      'depth_map',
                      'acc_map',
                      'ray_o',
                      ]

    def multi_scatter_zeros(ret):
        for key in expanding_keys:
            if key in ret:
                if ret[key].ndim == 2:
                    ret[key] = multi_scatter_(ret[key].new_zeros(*ray_d.shape[:-1]), inds, ret[key], dim=-1)
                elif ret[key].ndim == 3:
                    ret[key] = multi_scatter_(ret[key].new_zeros(*ray_d.shape[:-1], ret[key].shape[-1]), inds, ret[key], dim=-2)
        return ret

    # sphere tracing renderer will always use geometry normal
    if not training:
        if 'cpts' in locals():
            ret.cpts_map = cpts
        if 'bpts' in locals():
            ret.bpts_map = bpts
        if 'resd' in locals():
            ret.resd_map = resd
        if 'norm' in locals():
            ret.norm_map = norm
        if 'albedo' in locals():
            ret.albedo_map = albedo
        if 'roughness' in locals():
            ret.roughness_map = roughness[..., 0]  # to conform to the convention of outputting without last dimension TODO: this is ugly
        if not cfg.vis_rendering_map and not cfg.vis_shading_map and not cfg.vis_specular_map:
            ret = multi_scatter_zeros(ret)
            ret = alpha_output_(ret.acc_map, ret)  # return after blending results
            return ret  # early stop for visualization purpose

    # sphere tracing renderer will always use geometry visibility from DFSS
    if cfg.relighting:
        # possibly render light visibility # this is where most of the computation lies
        # heavy lifting: lower iter cnt, normally 8, but 512 times more query
        lvis, ldot = light_visibility(surf, norm, acc, xyz, sharp, shadow_tracing_decoder, bbox, **cfg.obj_lvis)

        # possibly pass through the rendering equation
        # light = expand_envmap_probe(envmap.probe, surf)  # B, eH, eW, P, 3
        surf2light = normalize(xyz[:, :, None] - surf[:, None, None])
        surf2cam = normalize(ray_o - surf)
        light = sample_envmap_image(envmap.probe, surf2light)  # B, eH, eW, P, 3
        # TODO: should we perform sampling instead of direct rendering here?
        # the gen_light_xyz function corresponding to a particular environment map is off a little
        if cfg.only_visibility:
            # debugging option
            ldot = torch.ones_like(ldot)  # make the final image look uniform without visibility
            light = light.mean(dim=-1, keepdim=True)  # single channel light for visibility
        if microfacet.cancel_cosine:
            ori_ldot = ldot
            ldot = torch.ones_like(ldot)
        shade = evaluate_shade(lvis, ldot, area, light)  # memory?
        brdf = microfacet(surf2light, surf2cam, norm, albedo, roughness)
        rgb = brdf * shade
        rgb = rgb.sum(dim=1).sum(dim=1)
        if cfg.tonemapping_rendering:
            rgb = linear2srgb(rgb)
        ret.rgb_map = rgb  # add before timing with acc

        # be slim during training
        if not training:

            # for visualization
            if cfg.vis_specular_map:
                spec_brdf = microfacet(surf2light, surf2cam, norm, 0.0, roughness)
                if microfacet.cancel_cosine:
                    # ignore these to make it more visible
                    ldot = 1 / (torch.abs(ldot) + 1e-8)
                else:
                    ldot = torch.ones_like(ldot)
                spec_shade = evaluate_shade(torch.ones_like(lvis), ldot, area, light)  # memory?
                rgb = spec_brdf * spec_shade
                rgb = rgb.sum(dim=1).sum(dim=1)
                ret.spec_map = rgb  # add before timing with acc

            # http://www.joshbarczak.com/blog/?p=272
            ldot = ldot if 'ori_ldot' not in locals() else ori_ldot
            shade = evaluate_shade(lvis, ldot, area, light)  # memory?
            shade = shade.sum(1).sum(1) * cfg.shading_albedo / np.pi  # brdf should sum to 1
            ret.shade_map = shade
            if cfg.vis_lvis_map: ret.shade_map = lvis.mean(1).mean(1)[..., None].expand(-1, -1, 3)
            if cfg.vis_ldot_map: ret.shade_map = ldot.mean(1).mean(1)[..., None].expand(-1, -1, 3)
            if cfg.vis_novel_light:
                ret.lvis_map = lvis.view(B, prod(lvis.shape[1:3]), lvis.shape[-1]).permute(0, 2, 1)  # MARK: MEM, DIM
                ret.ldot_map = ldot.view(B, prod(ldot.shape[1:3]), ldot.shape[-1]).permute(0, 2, 1)  # MARK: MEM, DIM

    else:
        ret.rgb_map = rgb

    if cfg.check_termination_sdf:
        if 'sdf_sum' not in globals():
            global sdf_sum, sdf_num
            sdf_sum = 0
            sdf_num = 0
        old_smpl_distance = cfg.smpl_distance
        cfg.smpl_distance = False
        d = sdf_decoder(surf)
        cfg.smpl_distance = old_smpl_distance
        cur_sum = d.abs().sum().item()
        cur_num = d.numel()
        sdf_sum += cur_sum
        sdf_num += cur_num
        print(f'avg sdf abs: {sdf_sum / sdf_num:.8f}')

    ret = multi_scatter_zeros(ret)
    # for k, v in ret.items():
    #     if v.isnan().any() or v.isinf().any():
    #         breakpoint()
    return ret


def render_bruteforce_human(ray_o: torch.Tensor,
                            ray_d: torch.Tensor,
                            near: torch.Tensor,
                            far: torch.Tensor,
                            xyz: torch.Tensor,
                            area: torch.Tensor,
                            envmap: dotdict,
                            microfacet: Microfacet,
                            training: bool,
                            ray_marching: Callable[[torch.Tensor], torch.Tensor],
                            net_decoder: Callable[[torch.Tensor], torch.Tensor],
                            sdf_decoder: Callable[[torch.Tensor], torch.Tensor],
                            norm_geometry: Callable[[torch.Tensor], torch.Tensor],
                            lvis_geometry: Callable[[torch.Tensor], torch.Tensor],
                            ):
    # ray marching surface intersection
    st, occ = ray_marching(ray_o, ray_d, near, far)  # B, P, 1; B, P, 1
    surf = ray_o + st * ray_d
    depth_map = (surf[..., 0] - ray_o[..., 0]) / ray_d[..., 0]  # MARK: CHANNEL
    acc = occ[..., 0]  # MARK: CHANNEL & GRADIENT

    # network forwarding
    ret = net_decoder(surf, ray_d, 0.005)
    raw = ret.raw
    ret.acc_map = acc
    if not training:
        ret.ray_o = ray_o
        ret.surf_map = surf
        ret.depth_map = depth_map

    # expand network output
    B, P, _ = ray_o.shape
    eH, eW = cfg.env_h, cfg.env_w

    if raw.shape[-1] == 3 + 1 + 3 + eH * eW + 1:  # is relighting (will not return rgb, needs rendering)
        albedo, roughness, norm, lvis, occ = raw.split([3, 1, 3, eH * eW, 1], dim=-1)  # B, P, X
    elif raw.shape[-1] == 3 + 3 + 1:
        norm, rgb, occ = raw.split([3, 1, 3])
    elif raw.shape[-1] == 3 + 3 + 3 + 3 + 3 + 1:
        cpts, bpts, resd, norm, rgb, occ = raw.split([3, 3, 3, 3, 3, 1], dim=-1)
    elif raw.shape[-1] == 3 + 1:
        # another type of network output, no need to explicitly render
        rgb, occ = raw.split([3, 1], dim=-1)
    elif raw.shape[-1] == 3 + 1 + 3 + 1:  # is relighting (will not return rgb, needs rendering)
        # expand network output
        albedo, roughness, norm, occ = raw.split([3, 1, 3, 1], dim=-1)  # B, P, X
    elif raw.shape[-1] == 3 + 3 + 3 + 3 + 1 + 3 + 1:
        cpts, bpts, resd, albedo, roughness, norm, occ = raw.split([3, 3, 3, 3, 1, 3, 1], dim=-1)  # B, P, X
    elif raw.shape[-1] == 3 + 3 + 3 + 3 + 3 + 1:
        cpts, bpts, resd, norm, rgb, occ = raw.split([3, 3, 3, 3, 3, 1], dim=-1)
    elif raw.shape[-1] == 3 + 3 + 1:
        norm, rgb, occ = raw.split([3, 3, 1], dim=-1)
    else:
        raise NotImplementedError(f'Supported ret.raw shape: {ret.raw.shape}')

    if not training and cfg.rgb_as_albedo:
        if 'albedo' in locals():
            albedo = srgb2linear(rgb)

    # for better visualization
    if not training and cfg.albedo_multiplier > 0:
        if 'albedo' in locals():
            albedo = albedo * cfg.albedo_multiplier

    if not training and cfg.zero_roughness:
        roughness = torch.zeros_like(occ)

    if not training and cfg.geometry_normal:
        norm = norm_geometry(surf)

    if not training:
        if 'cpts' in locals():
            ret.cpts_map = cpts
        if 'bpts' in locals():
            ret.bpts_map = bpts
        if 'resd' in locals():
            ret.resd_map = resd
        if 'norm' in locals():
            ret.norm_map = norm
        if 'albedo' in locals():
            ret.albedo_map = albedo
        if 'roughness' in locals():
            ret.roughness_map = roughness[..., 0]  # to conform to the convention of outputting without last dimension TODO: this is ugly
        if not cfg.vis_rendering_map and not cfg.vis_shading_map:
            ret = alpha_output_(ret.acc_map, ret)  # return after blending results
            return ret  # early stop for visualization purpose

    if not training and cfg.geometry_visibility:
        lvis = lvis_geometry(surf, norm)

    if cfg.relighting:
        lvis = lvis.permute(0, 2, 1).view(B, eH, eW, P)
        ldir = normalize(xyz)[None, :, :, None]
        ldot = (ldir * norm[:, None, None]).sum(dim=-1)  # B, eH, eW, P

        # possibly pass through the rendering equation
        surf2light = normalize(xyz[:, :, None] - surf[:, None, None])
        surf2cam = normalize(ray_o - surf)
        light = sample_envmap_image(envmap.probe, surf2light)  # B, eH, eW, P, 3
        if microfacet.cancel_cosine:
            ori_ldot = ldot
            ldot = torch.ones_like(ldot)
        shade = evaluate_shade(lvis, ldot, area, light)  # memory?
        brdf = evaluate_brdf(surf, albedo, roughness, norm, ray_o, xyz, microfacet)
        rgb = brdf * shade
        rgb = rgb.sum(dim=1).sum(dim=1)
        if cfg.tonemapping_rendering:
            rgb = linear2srgb(rgb)
        ret.rgb_map = rgb  # add before timing with acc

        # be slim during training
        if not training:

            # for visualization
            if cfg.vis_specular_map:
                spec_brdf = evaluate_brdf(surf, 0.0, roughness, norm, ray_o, xyz, microfacet)
                if microfacet.cancel_cosine:
                    # ignore these to make it more visible
                    ldot = 1 / (torch.abs(ldot) + 1e-8)
                else:
                    ldot = torch.ones_like(ldot)
                spec_shade = evaluate_shade(torch.ones_like(lvis), ldot, area, light)  # memory?
                rgb = spec_brdf * spec_shade
                rgb = rgb.sum(dim=1).sum(dim=1)
                ret.spec_map = rgb  # add before timing with acc

            ldot = ldot if 'ori_ldot' not in locals() else ori_ldot
            shade = shade.sum(1).sum(1) * cfg.shading_albedo / np.pi  # brdf should sum to 1
            ret.shade_map = shade
            if cfg.vis_novel_light:
                ret.lvis_map = lvis.view(B, prod(lvis.shape[1:3]), lvis.shape[-1]).permute(0, 2, 1)  # MARK: MEM, DIM
                ret.ldot_map = ldot.view(B, prod(ldot.shape[1:3]), ldot.shape[-1]).permute(0, 2, 1)  # MARK: MEM, DIM
    else:
        ret.rgb_map = rgb

    if cfg.check_termination_sdf:
        # filtered forward pass
        _, inds, _ = batch_aware_indexing(acc > 0.1, acc)  # MARK: SYNC
        surf = multi_gather(surf, inds)
        if 'sdf_sum' not in globals():
            global sdf_sum, sdf_num
            sdf_sum = 0
            sdf_num = 0
        old_smpl_distance = cfg.smpl_distance
        cfg.smpl_distance = False
        d = sdf_decoder(surf)
        cfg.smpl_distance = old_smpl_distance
        cur_sum = d.abs().sum().item()
        cur_num = d.numel()
        sdf_sum += cur_sum
        sdf_num += cur_num
        print(f'avg sdf abs: {sdf_sum / sdf_num:.8f}')

    return ret


class Renderer(nn.Module):
    def __init__(self, net: Network):
        super(Renderer, self).__init__()
        self.net = net

    def prepare_decoders(self, batch: dotdict):
        @net_chunkify
        def sdf_decoder(x: torch.Tensor, smooth_transition=True, **kwargs) -> dotdict: return self.net.inference_world_distance_field(x, batch, smooth_transition=smooth_transition, *kwargs)
        def shadow_sdf_decoder(x: torch.Tensor, smooth_transition=True, **kwargs) -> dotdict: return self.net.inference_world_distance_field(x, batch, smooth_transition=smooth_transition, **kwargs)  # MARK: Intricate config
        def net_decoder(x: torch.Tensor, v: torch.Tensor) -> dotdict: return self.net(x, v, 0.005, batch)  # full network forward pass (possibly other network type)

        @net_chunkify
        def blended_observed_sdf_decoder(x: torch.Tensor, smooth_transition=True, filtering=True, **kwargs) -> torch.Tensor: return self.net.inference_observed_distance_field(x, batch, smooth_transition=smooth_transition, filtering=filtering, **kwargs)
        @net_chunkify
        def observed_sdf_decoder(x: torch.Tensor, **kwargs) -> torch.Tensor: return self.net.inference_observed_distance_field(x, batch, **kwargs)
        @net_chunkify
        def world_to_can(x: torch.Tensor) -> torch.Tensor: return self.net.world_to_bigpose_transform(x, batch)
        @net_chunkify
        def can_to_world(x: torch.Tensor) -> torch.Tensor: return self.net.bigpose_to_world_transform(x, batch)

        def sphere_tracing_decoder(ray_o, ray_d, near, far, hdq_decoder, *args, **kwargs):
            return sphere_tracing(ray_o, ray_d, near, far,
                                  hdq_decoder if cfg.ablate_hdq_mode == 'hdq' else observed_sdf_decoder,
                                  world_to_can,
                                  can_to_world,
                                  mode=cfg.ablate_hdq_mode,  # when this is hdq, nothing should happen
                                  *args, **kwargs
                                  )

        def surface_tracing_decoder(ray_o, ray_d, near, far, *args, **kwargs):
            return sphere_tracing_decoder(ray_o, ray_d, near, far, sdf_decoder, *args, **kwargs)

        def shadow_tracing_decoder(ray_o, ray_d, near, far, *args, **kwargs):
            # TODO: whether inner penumbra is physically correct? without any actual hard shadow?
            return sphere_tracing_decoder(ray_o, ray_d, near, far, shadow_sdf_decoder, *args, **kwargs)  # different shadow tracing algorithm

        return surface_tracing_decoder, shadow_tracing_decoder, sdf_decoder, net_decoder

    def get_pixel_value(self,
                        ray_o: torch.Tensor,
                        ray_d: torch.Tensor,
                        near: torch.Tensor,
                        far: torch.Tensor,
                        envmap: dotdict,
                        batch: dotdict,
                        ) -> dotdict:

        if cfg.bruteforce_st:
            @chunkify(cfg.network_chunk_size // 64, dim=-2, merge_dims=True)
            def ray_marching(*args, **kwargs): return self.net.ray_marching(*args, **kwargs, batch=batch)
            @net_chunkify
            def net_decoder(*args, **kwargs): return self.net(*args, **kwargs, batch=batch)
            @net_chunkify
            def sdf_decoder(*args, **kwargs): return self.net.inference_world_distance_field(*args, **kwargs, batch=batch)
            @net_chunkify
            def geometry_normal(*args, **kwargs): return self.net.geometry_normal(*args, **kwargs, batch=batch)
            @net_chunkify
            def geometry_visibility(*args, **kwargs): return self.net.geometry_visibility(*args, **kwargs, batch=batch)

            return render_bruteforce_human(ray_o,
                                           ray_d,
                                           near,
                                           far,
                                           self.net.light_xyz if hasattr(self.net, 'light_xyz') else None,
                                           self.net.light_area if hasattr(self.net, 'light_area') else None,
                                           envmap,
                                           self.net.microfacet if hasattr(self.net, 'microfacet') else None,
                                           self.net.training if hasattr(self.net, 'training') else None,
                                           ray_marching,
                                           net_decoder,
                                           sdf_decoder,
                                           geometry_normal,
                                           geometry_visibility,
                                           )
        else:
            surface_tracing_decoder, shadow_tracing_decoder, sdf_decoder, net_decoder = self.prepare_decoders(batch)
            # unable to batch this
            bbox = batch.wbounds
            bbox[:, 0] -= cfg.env_lvis.bbox_margin
            bbox[:, 1] += cfg.env_lvis.bbox_margin

            return render_human(ray_o,
                                ray_d,
                                near,
                                far,
                                envmap,
                                self.net.light_xyz if hasattr(self.net, 'light_xyz') else None,
                                self.net.light_area if hasattr(self.net, 'light_area') else None,
                                self.net.light_sharp if hasattr(self.net, 'light_sharp') else None,
                                self.net.training if hasattr(self.net, 'training') else None,
                                self.net.microfacet if hasattr(self.net, 'microfacet') else None,
                                bbox,
                                surface_tracing_decoder,
                                shadow_tracing_decoder,
                                sdf_decoder,  # for training geometry (passing gradients in)
                                net_decoder,  # for evaluating material properties
                                )

    def get_ground_value(self,
                         ray_o: torch.Tensor,
                         ray_d: torch.Tensor,
                         acc: torch.Tensor,
                         envmap: dotdict,
                         batch: dotdict,
                         ) -> dotdict:

        surface_tracing_decoder, shadow_tracing_decoder, sdf_decoder, net_decoder = self.prepare_decoders(batch)

        # unable to batch this
        bbox = batch.wbounds
        bbox[:, 0] -= cfg.env_lvis.bbox_margin
        bbox[:, 1] += cfg.env_lvis.bbox_margin
        return render_ground(ray_o,
                             ray_d,
                             acc,
                             self.net.light_xyz,
                             self.net.light_area,
                             self.net.light_sharp,
                             bbox,
                             envmap,
                             shadow_tracing_decoder,
                             )

    def render(self, batch):
        # for now, maybe skip light evaluation (not used anyway in monosdf setup)
        if not self.net.training and cfg.replace_light:
            envmap = batch.novel_lights[cfg.replace_light]
        elif hasattr(self.net, 'global_env_map'):
            envmap = dotdict(probe=self.net.global_env_map[None])  # optimizable, with a fake batch dimension
        else:
            envmap = None
        # if 'iter_step' in batch.meta and batch.meta.iter_step == 5001: breakpoint()
        # actual network evaluations
        @chunkify(cfg.render_chunk_size, dim=-2, print_progress=cfg.print_render_progress)
        def chunk_pixel_value(*args, **kwargs): return self.get_pixel_value(*args, **kwargs)
        ret = chunk_pixel_value(batch.ray_o, batch.ray_d, batch.near, batch.far, envmap, batch)

        # post processing
        ret.envmap = envmap  # later used for adding light probe

        # possibly return shade map or add ground rendering result
        if not self.net.training and cfg.vis_ground_shading:
            # preparing input values for ground shading
            B, P, _ = batch.ray_o.shape
            H, W = batch.meta.H.item(), batch.meta.W.item()
            F = H * W
            spec_mab = batch.mask_at_box.view(B, F).clone()  # avoid being changed with some side-effects
            _, inds, _ = batch_aware_indexing(spec_mab)  # MARK: SYNC
            ray_o, ray_d = get_rays(H, W, batch.cam_K, batch.cam_R, batch.cam_T)
            ray_o, ray_d = ray_o.view(B, F, 3), ray_d.view(B, F, 3)
            acc = multi_scatter_(torch.ones_like(spec_mab, dtype=torch.float), inds, 1 - ret.acc_map, dim=-1)  # B, F

            # actual network evaluations
            @chunkify(cfg.render_chunk_size, dim=-2, print_progress=cfg.print_render_progress)
            def chunk_ground_value(*args, **kwargs): return self.get_ground_value(*args, **kwargs)
            ground = chunk_ground_value(ray_o, ray_d, acc, envmap, batch)

            # preparing values for blending or visualization
            ground.ray_o = ray_o  # later used for blending
            ground.ray_d = ray_d  # later used for blending
            ground.acc_map = acc  # later used for blending
            ground.inds = inds  # later used for blending
            batch.mask_at_box[:] = True  # later used for visualization

            # maybe blend foreground and background rendering results
            if cfg.vis_novel_light:  # prepare for visualization
                ret.ground = ground
            else:
                ret = blend_output_(ground.acc_map, ground.inds, ground, ret)
        else:
            ret = alpha_output_(ret.acc_map, ret)

        return ret
