# Main
import os
import math
import torch
import random
import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from termcolor import colored
from itertools import accumulate
from functools import lru_cache, reduce
from torch.distributed.optim import ZeroRedundancyOptimizer
from smplx.lbs import batch_rodrigues, batch_rigid_transform

# Typing
from typing import List, Callable, Tuple

# Utils
from lib.utils.log_utils import log
from lib.utils.base_utils import dotdict


def schlick_bias(x, s): return (s * x) / ((s - 1) * x + 1)


def schlick_gain(x, s): return torch.where(x < 0.5, schlick_bias(2 * x, s) / 2, (schlick_bias(2 * x - 1, 1 - s) + 1) / 2)


def resize_image(img: torch.Tensor, uH, uW, mode='area'):
    sh = img.shape
    if len(sh) == 4 and sh[-1] == 3:  # assumption
        img = img.permute(0, 3, 1, 2)
    elif len(sh) == 3 and sh[-1] == 3:  # assumption
        img = img.permute(2, 0, 1)[None]
    if mode == 'bilinear' or mode == 'bicubic':
        img = F.interpolate(img, size=(uH, uW), mode=mode, align_corners=False)  # uH, uW, 3
    else:
        img = F.interpolate(img, size=(uH, uW), mode=mode)  # uH, uW, 3
    if len(sh) == 4 and sh[-1] == 3:
        img = img.permute(0, 2, 3, 1)
    elif len(sh) == 3 and sh[-1] == 3:  # assumption
        img = img[0].permute(1, 2, 0)
    return img


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    if x.ndim == xp.ndim - 1:
        x = x[None]

    m = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1] + torch.finfo(xp.dtype).eps)  # slope
    b = fp[..., :-1] - (m * xp[..., :-1])

    indices = torch.sum(torch.ge(x[..., :, None], xp[..., None, :]), -1) - 1  # torch.ge:  x[i] >= xp[i] ? true: false
    indices = torch.clamp(indices, 0, m.shape[-1] - 1)

    return m.gather(dim=-1, index=indices) * x + b.gather(dim=-1, index=indices)


def integrate_weights(w: torch.Tensor):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.
    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.
    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.
    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = torch.cumsum(w[..., :-1], dim=-1).clip(max=1.0)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = torch.cat([cw.new_zeros(shape), cw, cw.new_ones(shape)], dim=-1)
    return cw0


def weighted_percentile(t: torch.Tensor, w: torch.Tensor, ps: list):
    """Compute the weighted percentiles of a step function. w's must sum to 1."""
    t, w = matchup_channels(t, w)
    cw = integrate_weights(w)
    # We want to interpolate into the integrated weights according to `ps`.
    # Vmap fn to an arbitrary number of leading dimensions.
    cw_mat = cw.reshape([-1, cw.shape[-1]])
    t_mat = t.reshape([-1, t.shape[-1]])
    wprctile_mat = interpolate(torch.from_numpy(np.array(ps)).to(t, non_blocking=True),
                               cw_mat,
                               t_mat)
    wprctile = wprctile_mat.reshape(cw.shape[:-1] + (len(ps),))
    return wprctile


def ray_transfer(s: torch.Tensor,
                 tn: torch.Tensor,
                 tf: torch.Tensor,
                 g: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                 ig: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                 ):
    # transfer ray depth from s space to t space (with inverse of g)
    return ig(s * g(tf) + (1 - s) * g(tn))


def inv_transfer(t: torch.Tensor,
                 tn: torch.Tensor,
                 tf: torch.Tensor,
                 g: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                 ):
    # transfer ray depth from t space back to s space (with function g)
    return (g(t) - g(tn)) / (g(tf) - g(tn))

# implement the inverse distance sampling stragety of mipnerf360


def linear_sampling(device='cuda',
                    n_samples: int = 128,
                    perturb=False,
                    ):
    # calculate the steps for each ray
    s_vals = torch.linspace(0., 1. - 1 / n_samples, steps=n_samples, device=device)  # S,
    if perturb:
        s_vals = s_vals + torch.rand_like(s_vals) / n_samples  # S,
    return s_vals

# Hierarchical sampling (section 5.2)


def searchsorted(a: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find indices where v should be inserted into a to maintain order.
    This behaves like jnp.searchsorted (its second output is the same as
    jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
    faster because it wastes memory to save some compute.
    Args:
      a: tensor, the sorted reference points that we are scanning to see where v
        should lie.
      v: tensor, the query points that we are pretending to insert into a. Does
        not need to be sorted. All but the last dimensions should match or expand
        to those of a, the last dimension can differ.
    Returns:
      (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
      range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
      last index of a.
    """
    i = torch.arange(a.shape[-1], device=a.device)  # 128
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = torch.max(torch.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2)[0]  # 128
    idx_hi = torch.min(torch.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2)[0]
    return idx_lo, idx_hi


def invert_cdf(u, t, w):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    cw = integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = interpolate(u, cw, t)
    return t_new


def importance_sampling(t: torch.Tensor,
                        w: torch.Tensor,
                        num_samples: int,
                        perturb=True,
                        single_jitter=False,
                        ):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
        rng: random number generator (or None for `linspace` sampling).
        t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
        w_logits: [..., num_bins], logits corresponding to bin weights
        num_samples: int, the number of samples.
        single_jitter: bool, if True, jitter every sample along each ray by the same
        amount in the inverse CDF. Otherwise, jitter each sample independently.
        deterministic_center: bool, if False, when `rng` is None return samples that
        linspace the entire PDF. If True, skip the front and back of the linspace
        so that the centers of each PDF interval are returned.
        use_gpu_resampling: bool, If True this resamples the rays based on a
        "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
        this resamples the rays based on brute-force searches, which is fast on
        TPUs, but slow on GPUs.

    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """

    # preparing for size change
    sh = *t.shape[:-1], num_samples  # B, P, I
    t = t.reshape(-1, t.shape[-1])
    w = w.reshape(-1, w.shape[-1])

    # assuming sampling in s space
    if t.shape[-1] != w.shape[-1] + 1:
        t = torch.cat([t, torch.ones_like(t[..., -1:])], dim=-1)

    eps = torch.finfo(torch.float32).eps

    # Draw uniform samples.

    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u_max = eps + (1 - eps) / num_samples
    max_jitter = (1 - u_max) / (num_samples - 1) - eps if perturb else 0
    d = 1 if single_jitter else num_samples
    u = (
        torch.linspace(0, 1 - u_max, num_samples, device=t.device) +
        torch.rand(t.shape[:-1] + (d,), device=t.device) * max_jitter
    )

    u = invert_cdf(u, t, w)

    # preparing for size change
    u = u.reshape(sh)
    return u


def matchup_channels(t: torch.Tensor, w: torch.Tensor):
    if t.shape[-1] != w.shape[-1] + 1:
        t = torch.cat([t, torch.ones_like(t[..., -1:])], dim=-1)  # 65
    return t, w


def weight_to_pdf(t: torch.Tensor, w: torch.Tensor, eps=torch.finfo(torch.float32).eps**2):
    t, w = matchup_channels(t, w)
    """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
    return w / (t[..., 1:] - t[..., :-1]).clip(eps)


def pdf_to_weight(t: torch.Tensor, p: torch.Tensor):
    t, p = matchup_channels(t, p)
    """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
    return p * (t[..., 1:] - t[..., :-1])


def max_dilate(t, w, dilation, domain=(-torch.inf, torch.inf)):
    t, w = matchup_channels(t, w)
    """Dilate (via max-pooling) a non-negative step function."""
    t0 = t[..., :-1] - dilation
    t1 = t[..., 1:] + dilation
    t_dilate = torch.sort(torch.cat([t, t0, t1], dim=-1), dim=-1)[0]
    t_dilate = t_dilate.clip(*domain)
    w_dilate = torch.max(
        torch.where(
            (t0[..., None, :] <= t_dilate[..., None])
            & (t1[..., None, :] > t_dilate[..., None]),
            w[..., None, :],
            0,
        ),
        dim=-1)[0][..., :-1]
    return t_dilate, w_dilate


def max_dilate_weights(t,
                       w,
                       dilation,
                       domain=(-torch.inf, torch.inf),
                       renormalize=False,
                       eps=torch.finfo(torch.float32).eps**2):
    """Dilate (via max-pooling) a set of weights."""
    p = weight_to_pdf(t, w)
    t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
    w_dilate = pdf_to_weight(t_dilate, p_dilate)
    if renormalize:
        w_dilate /= torch.sum(w_dilate, dim=-1, keepdim=True).clip(eps)
    return t_dilate, w_dilate


def query(tq, t, y, outside_value=0):
    """Look up the values of the step function (t, y) at locations tq."""
    idx_lo, idx_hi = searchsorted(t, tq)
    yq = torch.where(idx_lo == idx_hi, outside_value,
                     torch.take_along_dim(torch.cat([y, torch.full_like(y[..., :1], outside_value)], dim=-1), idx_lo, dim=-1))  # ?
    return yq


def chunkify(chunk_size=8, key='img', pos=0, dim=0, merge_dims=False, print_progress=False):
    # will fail if dim == -1, currently only tested on dim == -2 or dim == 1
    # will select a key element from the argments: either by keyword `key` or position `pos`
    # then, depending on whether user wants to merge other dimensions, will select the dim to chunkify according to `dim`
    def wrapper(decoder: Callable[[torch.Tensor], torch.Tensor]):
        def decode(*args, **kwargs):
            # Prepare pivot args (find shape information from this arg)
            if key in kwargs:
                x: torch.Tensor = kwargs[key]
            else:
                x: torch.Tensor = args[pos]
                args = [*args]
            sh = x.shape[:dim + 1]  # record original shape up until the chunkified dim
            nn_dim = len(sh) - 1  # make dim a non-negative number (i.e. -2 to 1?)

            # Prepare all tensor arguments by filtering with isinstance
            tensor_args = [v for v in args if isinstance(v, torch.Tensor)]
            tensor_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
            other_args = [v for v in args if not isinstance(v, torch.Tensor)]
            other_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}

            # Merge all dims except first batch dim up until the actual chunkify dimension
            if merge_dims:
                x = x.view(x.shape[0], -1, *x.shape[nn_dim + 1:])
                tensor_args = [v.view(v.shape[0], -1, *v.shape[nn_dim + 1:]) for v in tensor_args]
                tensor_kwargs = {k: v.view(v.shape[0], -1, *v.shape[nn_dim + 1:]) for k, v in tensor_kwargs.items()}
                nn_dim = 1  # will always be 1 in this situation

            # Running the actual batchified forward pass
            ret = []
            total_size = x.shape[nn_dim]
            # We need to update chunk size so that almost all chunk has a decent amount of queries
            actual_size = math.ceil(total_size / math.ceil(total_size / chunk_size)) if total_size else chunk_size  # this value should be smaller than the actual chunk_size specified
            pbar = tqdm(total=total_size, disable=not print_progress)
            for i in range(0, total_size, actual_size):
                # nn_dim should be used if there's multiplication involved
                chunk_args = [v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for v in tensor_args]
                chunk_kwargs = {k: v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for k, v in tensor_kwargs.items()}
                ret.append(decoder(*chunk_args, *other_args, **chunk_kwargs, **other_kwargs))
                pbar.update(min(i + actual_size, total_size) - i)

            if not len(ret):
                # brute-forcely go through the network with empty input
                log(f'zero length tensor detected in chunkify, are the camera parameters correct?', 'red')
                i = 0
                chunk_args = [v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for v in tensor_args]
                chunk_kwargs = {k: v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for k, v in tensor_kwargs.items()}
                ret.append(decoder(*chunk_args, *other_args, **chunk_kwargs, **other_kwargs))

            # Merge ret list based on reture type (single tensor or dotdict?)
            # Return values of chunified function should all be tensors
            if len(ret) and isinstance(ret[0], torch.Tensor):
                ret = torch.cat(ret, dim=nn_dim)
                ret = ret.view(*sh, *ret.shape[nn_dim + 1:]) if x.shape[nn_dim] == ret.shape[nn_dim] else ret
            elif len(ret) and isinstance(ret[0], dict):
                dict_type = type(ret[0])
                ret = {k: torch.cat([v[k] for v in ret], dim=nn_dim) for k in ret[0].keys()}
                ret = {k: v.view(*sh, *v.shape[nn_dim + 1:]) if x.shape[nn_dim] == v.shape[nn_dim] else v for k, v in ret.items()}
                ret = dict_type(ret)
            elif len(ret) and (isinstance(ret[0], list) or isinstance(ret[0], tuple)):
                list_type = type(ret[0])
                ret = [torch.cat([v[i] for v in ret], dim=nn_dim) for i in range(len(ret[0]))]
                ret = list_type(ret)
            else:
                __import__('ipdb').set_trace()
                raise RuntimeError(f'Unsupported return type to batchify: {type(ret[0])}, or got empty return value')
            return ret
        return decode
    return wrapper


def key_cache(key: Callable):
    def key_cache_wrapper(func: Callable):
        # will only use argument that match the key positiona or name in the args or kwargs collection as lru_cache's key
        cached_result = None
        cached_hash = None

        def func_wrapper(*args, **kwargs):
            nonlocal cached_result, cached_hash
            key_value = key(*args, **kwargs)
            key_hash = hash(key_value)
            if key_hash != cached_hash:
                cached_result = func(*args, **kwargs)
                cached_hash = key_hash
            return cached_result

        return func_wrapper
    return key_cache_wrapper


def batch_aware_indexing(mask: torch.Tensor, metric: torch.Tensor = None, dim=-1) -> Tuple[torch.Tensor, torch.Tensor, int]:  # MARK: SYNC
    # dim: in terms of the index (mask)
    if mask.dtype != torch.bool: mask = mask.bool()
    if metric is None: metric = mask.int()
    if metric.dtype == torch.bool: metric = metric.int()
    # retain all other dimensions (likely batch dimensions)
    S = mask.sum(dim=dim).max().item()  # the max value of this dim on all other dimension
    valid, inds = metric.topk(S, dim=dim, sorted=False)  # only find the top (mask = True) values (randomly select other values)
    return valid, inds, S


def compute_ground_tris(o: torch.Tensor, d: torch.Tensor):
    n = normalize(torch.rand_like(d))  # B, P, 3,
    a = torch.cross(d, n)
    b = torch.cross(d, a)
    return torch.stack([o, o + a, o + b], dim=-1).mT  # B, P, 3, 3 (considering the right normal)


def torch_dot(x: torch.Tensor, y: torch.Tensor):
    return (x * y).sum(dim=-1)


def get_rays(H: int, W: int, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor, subpixel=False):
    # calculate the camera origin
    ray_o = -(R.mT @ T).ravel()
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(H, dtype=R.dtype, device=R.device),
                          torch.arange(W, dtype=R.dtype, device=R.device),
                          indexing='ij')
    # 0->H, 0->W
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=2)
    if subpixel:
        rand = torch.rand(H, W, 2, device=R.device, dtype=R.dtype) - 0.5
        xy1[:, :, :2] += rand
    pixel_camera = xy1 @ torch.inverse(K).mT
    pixel_world = (pixel_camera - T.ravel()) @ R
    # calculate the ray direction
    ray_o = ray_o[None, None].expand(pixel_world.shape)
    ray_d = normalize(pixel_world - ray_o)
    return ray_o, ray_d


def angle_to_rotation_2d(theta: torch.Tensor):
    sin = theta.sin()
    cos = theta.cos()
    R = theta.new_zeros(*theta.shape[:-1], 2, 2)
    R[..., 0, :1] = cos
    R[..., 1, :1] = sin
    R[..., 0, 1:] = -sin
    R[..., 1, 1:] = cos

    return R


def multi_indexing(index: torch.Tensor, shape: torch.Size, dim=-2):
    # index will first be augmented to match the values' dimentionality at the back
    # then we will try to broatcast index's shape to values shape
    shape = list(shape)
    back_pad = len(shape) - index.ndim
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)


def multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
    # index will first be augmented to match the values' dimentionality at the back
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the values's second to last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(index, values.shape, dim))


def multi_scatter(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=-2):
    # backward of multi_gather
    return target.scatter(dim, multi_indexing(index, values.shape, dim), values)


def multi_scatter_(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=-2):
    # inplace version of multi_scatter
    return target.scatter_(dim, multi_indexing(index, values.shape, dim), values)


def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1):
        f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3


def linear_indexing(index: torch.Tensor, shape: torch.Size, dim=0):
    assert index.ndim == 1
    shape = list(shape)
    dim = dim if dim >= 0 else len(shape) + dim
    front_pad = dim
    back_pad = len(shape) - dim - 1
    for _ in range(front_pad):
        index = index.unsqueeze(0)
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)


def linear_gather(values: torch.Tensor, index: torch.Tensor, dim=0):
    # only taking linea indices as input
    return values.gather(dim, linear_indexing(index, values.shape, dim))


def linear_scatter(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter(dim, linear_indexing(index, values.shape, dim), values)


def linear_scatter_(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter_(dim, linear_indexing(index, values.shape, dim), values)


def merge01(x: torch.Tensor):
    return x.reshape(-1, *x.shape[2:])


def scatter0(target: torch.Tensor, inds: torch.Tensor, value: torch.Tensor):
    return target.scatter(0, expand_at_the_back(target, inds), value)  # Surface, 3 -> B * S, 3


def gather0(target: torch.Tensor, inds: torch.Tensor):
    return target.gather(0, expand_at_the_back(target, inds))  # B * S, 3 -> Surface, 3


def expand_at_the_back(target: torch.Tensor, inds: torch.Tensor):
    for _ in range(target.ndim - 1):
        inds = inds.unsqueeze(-1)
    inds = inds.expand(-1, *target.shape[1:])
    return inds


def expand0(x: torch.Tensor, B: int):
    return x[None].expand(B, *x.shape)


def expand1(x: torch.Tensor, P: int):
    return x[:, None].expand(-1, P, *x.shape[1:])


def nonzero0(condition: torch.Tensor):
    # MARK: will cause gpu cpu sync
    # return those that are true in the provided tensor
    return condition.nonzero(as_tuple=True)[0]


def get_wsampling_points(ray_o: torch.Tensor, ray_d: torch.Tensor, wpts: torch.Tensor, z_interval=0.01, n_samples=11, perturb=True):
    # calculate the steps for each ray
    z_vals = torch.linspace(-z_interval, z_interval, steps=n_samples, dtype=ray_d.dtype, device=ray_d.device)

    if n_samples == 1:
        z_vals[:] = 0

    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, dtype=ray_d.dtype, device=ray_d.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = wpts[:, :, None] + ray_d[:, :, None] * z_vals[:, None]  # B, N, S, 3
    z_vals = (pts[..., 0] - ray_o[..., :1]) / ray_d[..., :1]  # using x dim to calculate depth

    return pts, z_vals

# Grouped convolution for SLRF


def grouped_mlp(I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU(), type='fused'):
    if type == 'fused':
        return FusedGroupedMLP(I, N, W, D, Z, actvn)  # fast, worse, # ? why: problem with grad magnitude
    elif type == 'gconv':
        return GConvGroupedMLP(I, N, W, D, Z, actvn)  # slow, better


class GradModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradModule, self).__init__()

    def take_gradient(self, output: torch.Tensor, input: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
        return take_gradient(output, input, d_out, self.training or create_graph, self.training or retain_graph)

    def jacobian(self, output: torch.Tensor, input: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [self.take_gradient(o, input, retain_graph=(i < len(outputs))) for i, o in enumerate(outputs)]
        jac = torch.stack(grads, dim=-1)
        return jac


class FusedGroupedMLP(GradModule):
    # I: input dim
    # N: group count
    # W: network width
    # D: network depth
    # Z: output dim
    # actvn: network activation

    # Fisrt layer: (B, N * I, S) -> (B * N, I, S) -> (B * N, S, I)
    # Weight + bias: (N, I, W) + (N, W) -> pad to (B, N, I, W) -> (B * N, I, W)
    # Result: (B * N, S, W) + (B * N, S, W)
    def __init__(self, I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU()):
        super(FusedGroupedMLP, self).__init__()
        self.N = N
        self.I = I
        self.Z = Z
        self.W = W
        self.D = D
        self.actvn = actvn

        self.Is = \
            [I] +\
            [W for _ in range(D - 2)] +\
            [W]\

        self.Zs = \
            [W] +\
            [W for _ in range(D - 2)] +\
            [Z]\

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(N, I, W))] +
            [nn.Parameter(torch.empty(N, W, W)) for _ in range(D - 2)] +
            [nn.Parameter(torch.empty(N, W, Z))]
        )

        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(N, W))] +
            [nn.Parameter(torch.empty(N, W)) for _ in range(D - 2)] +
            [nn.Parameter(torch.empty(N, Z))]
        )

        for i, w in enumerate(self.weights):  # list stores reference
            ksqrt = np.sqrt(1 / self.Is[i])
            nn.init.uniform_(w, -ksqrt, ksqrt)
        for i, b in enumerate(self.biases):
            ksqrt = np.sqrt(1 / self.Is[i])
            nn.init.uniform_(b, -ksqrt, ksqrt)

    def forward(self, x: torch.Tensor):
        B, N, S, I = x.shape
        x = x.view(B * self.N, S, self.I)

        for i in range(self.D):
            I = self.Is[i]
            Z = self.Zs[i]
            w = self.weights[i]  # N, W, W
            b = self.biases[i]  # N, W

            w = w[None].expand(B, -1, -1, -1).reshape(B * self.N, I, Z)
            b = b[None, :, None].expand(B, -1, -1, -1).reshape(B * self.N, -1, Z)
            x = torch.baddbmm(b, x, w)  # will this just take mean along batch dimension?
            if i < self.D - 1:
                x = self.actvn(x)

        x = x.view(B, self.N, S, self.Z)  # ditching gconv

        return x


class GConvGroupedMLP(GradModule):
    def __init__(self, I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU()):
        # I: input dim
        # N: group count
        # W: network width
        # D: network depth
        # Z: output dim
        # actvn: network activation
        super(GConvGroupedMLP, self).__init__()
        self.mlp = nn.ModuleList(
            [nn.Conv1d(N * I, N * W, 1, groups=N), actvn] +
            [f for f in [nn.Conv1d(N * W, N * W, 1, groups=N), actvn] for _ in range(D - 2)] +
            [nn.Conv1d(N * W, N * Z, 1, groups=N)]
        )
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: torch.Tensor):
        # x: B, N, S, C
        B, N, S, I = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B, N * I, S)
        x = self.mlp(x)
        x = x.reshape(B, N, -1, S)
        x = x.permute(0, 1, 3, 2)
        return x


class cVAE(GradModule):  # group cVAE with grouped convolution
    def __init__(self,
                 group_cnt: int,
                 latent_dim: int,
                 in_dim: int,
                 cond_dim: int,
                 out_dim: int,

                 encode_w: int,
                 encode_d: int,

                 decode_w: int,
                 decode_d: int,
                 ):
        super(cVAE, self).__init__()

        self.N = group_cnt
        self.L = latent_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # input: embedded time concatenated with multiplied pose
        # output: mu and log_var
        I = in_dim + cond_dim
        N, W, D, Z = group_cnt, encode_w, encode_d, latent_dim * 2
        self.encoder = grouped_mlp(I, N, W, D, Z)

        # input: reparameterized latent variable
        # output: high-dim embedding + 3D residual node trans
        I = latent_dim + cond_dim
        N, W, D, Z = group_cnt, decode_w, decode_d, out_dim
        self.decoder = grouped_mlp(I, N, W, D, Z)

    def encode(self, x: torch.Tensor):
        # x: B, N, S, I
        mu, log_var = self.encoder(x).split([self.L, self.L], dim=-1)
        return mu, log_var

    def decode(self, z: torch.Tensor):
        # z: B, N, S, 8
        out = self.decoder(z)
        return out

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        input_ndim = x.ndim
        if input_ndim == 3 and self.N == 1:
            x = x[:, None]
            c = c[:, None]
        elif input_ndim == 2 and self.N == 1:
            x = x[:, None, None]
            c = c[:, None, None]
        else:
            raise NotImplementedError(f'Unsupported input shape: x.shape: {x.shape}, c.shape: {c.shape} for node count: {self.N}')

        # x: B, N, S, C, where C is N * in_dim
        # where in_dim should be embedded time concatenated with multiplied pose
        mu, log_var = self.encode(torch.cat([x, c], dim=-1))  # this second is a lot slower than decode, why?
        z = self.reparameterize(mu, log_var)
        out = self.decode(torch.cat([z, c], dim=-1))
        # out: B, N, S, out_dim(1)
        # mu: B, N, S, 8, log_var: B, N, S, 8, z: B, N, S, 8

        if input_ndim == 3 and self.N == 1:
            out = out[:, 0]
            mu = mu[:, 0]
            log_var = log_var[:, 0]
            z = z[:, 0]
        elif input_ndim == 2 and self.N == 1:
            out = out[:, 0, 0]
            mu = mu[:, 0, 0]
            log_var = log_var[:, 0, 0]
            z = z[:, 0, 0]
        else:
            raise NotImplementedError(f'Unsupported input shape: x.shape: {x.shape}, c.shape: {c.shape} for node count: {self.N}')

        return out, mu, log_var, z

# Resnet Blocks


class ResnetBlock(nn.Module):
    """
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, kernel_size, size_out=None, size_h=None):
        super(ResnetBlock, self).__init__()

        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        padding = kernel_size // 2
        self.conv_0 = nn.Conv2d(size_in, size_h, kernel_size=kernel_size, padding=padding)
        self.conv_1 = nn.Conv2d(size_h, size_out, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(size_in, size_out, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        net = self.conv_0(self.activation(x))
        dx = self.conv_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def number_of_params(network: nn.Module):
    return sum([p.numel() for p in network.parameters() if p.requires_grad])


def make_params(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=True)


def make_buffer(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=False)


def raw2outputs(raw, z_vals, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = raw[..., :-1]  # [n_rays, n_samples, 3]
    alpha = raw[..., -1]  # [n_rays, n_samples]

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1), dtype=alpha.dtype, device=alpha.device), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [n_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, dtype=depth_map.dtype, device=depth_map.device),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def raw2alpha(raw: torch.Tensor, dists=0.005, bias=0.0, act_fn=F.relu):
    if isinstance(dists, torch.Tensor):
        if dists.ndim == raw.ndim - 1:
            dists = dists[..., None]
    return 1. - torch.exp(-act_fn(raw + bias) * dists)


def alpha2raw(alpha, dists=0.005, bias=0.0, act_fn=F.relu):
    return act_fn(-torch.log(1 - alpha) / dists) - bias


def alpha2sdf(alpha, beta, dists=0.005):
    return beta * torch.log(2 * beta * (-torch.log(1 - alpha) / dists))


def sdf_to_occ(sdf: torch.Tensor, beta: torch.Tensor, dists=0.005):
    sigma = sdf_to_sigma(sdf, beta)
    occ = raw2alpha(sigma, dists)
    return occ


# @torch.jit.script  # will fuse element wise operations together to make a faster invokation
def compute_val0(x: torch.Tensor, beta: torch.Tensor, ind0: torch.Tensor):
    # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
    val0 = 1 / beta * (0.5 * (x * ind0 / beta).exp()) * ind0
    return val0


# @torch.jit.script  # will fuse element wise operations together to make a faster invokation
def compute_val1(x: torch.Tensor, beta: torch.Tensor, ind1: torch.Tensor):
    # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
    val1 = 1 / beta * (1 - 0.5 * (-x * ind1 / beta).exp()) * ind1
    return val1


def sdf_to_sigma(sdf: torch.Tensor, beta: torch.Tensor):
    # double the computation, but no synchronization needed
    x = -sdf
    ind0 = x <= 0
    ind1 = ~ind0

    return compute_val0(x, beta, ind0) + compute_val1(x, beta, ind1)


def torch_unique_with_indices_and_inverse(x, dim=0):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    indices, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, indices.new_empty(unique.size(dim)).scatter_(dim, indices, perm), inverse


def unmerge_faces(faces: torch.Tensor, *args):
    # stack into pairs of (vertex index, texture index)
    stackable = [faces.reshape(-1)]
    # append multiple args to the correlated stack
    # this is usually UV coordinates (vt) and normals (vn)
    for arg in args:
        stackable.append(arg.reshape(-1))

    # unify them into rows of a numpy array
    stack = torch.column_stack(stackable)
    # find unique pairs: we're trying to avoid merging
    # vertices that have the same position but different
    # texture coordinates
    _, unique, inverse = torch_unique_with_indices_and_inverse(stack)

    # only take the unique pairs
    pairs = stack[unique]
    # try to maintain original vertex order
    order = pairs[:, 0].argsort()
    # apply the order to the pairs
    pairs = pairs[order]

    # we re-ordered the vertices to try to maintain
    # the original vertex order as much as possible
    # so to reconstruct the faces we need to remap
    remap = torch.zeros(len(order), dtype=torch.long, device=faces.device)
    remap[order] = torch.arange(len(order), device=faces.device)

    # the faces are just the inverse with the new order
    new_faces = remap[inverse].reshape((-1, 3))

    # the mask for vertices and masks for other args
    result = [new_faces]
    result.extend(pairs.T)

    return result


def merge_faces(faces, *args, n_verts=None):
    # TODO: batch this
    # remember device the faces are on
    device = faces.device
    # start with not altering faces at all
    result = [faces]
    # find the maximum index referenced by faces
    if n_verts is None:  # sometimes things get padded
        n_verts = faces.max() + 1
    # add a vertex mask which is just ordered
    result.append(torch.arange(n_verts, device=device))

    # now given the order is fixed do our best on the rest of the order
    for arg in args:
        # create a mask of the attribute-vertex mapping
        # note that these might conflict since we're not unmerging
        masks = torch.zeros((3, n_verts), dtype=torch.long, device=device)
        # set the mask using the unmodified face indexes
        for i, f, a in zip(range(3), faces.permute(*torch.arange(faces.ndim - 1, -1, -1)), arg.permute(*torch.arange(arg.ndim - 1, -1, -1))):
            masks[i][f] = a
        # find the most commonly occurring attribute (i.e. UV coordinate)
        # and use that index note that this is doing a float conversion
        # and then median before converting back to int: could also do this as
        # a column diff and sort but this seemed easier and is fast enough
        result.append(torch.median(masks, dim=0)[0].to(torch.long))

    return result


def volume_rendering(rgb, alpha, eps=1e-8, bg_brightness=0.0, bg_image=None):
    # NOTE: here alpha's last dim is not 1, but n_samples
    # rgb: n_batch, n_rays, n_samples, 3
    # alpha: n_batch, n_rays, n_samples
    # bg_image: n_batch, n_rays, 3 or None, if this is given as not None, the last sample on the ray will be replaced by this value (assuming this lies on the background)
    # We need to assume:
    # 1. network will find the True geometry, thus giving the background image its real value
    # 2. background image is rendered in a non-static fasion
    # returns:
    # weights: n_batch, n_rays, n_samples
    # rgb_map: n_batch, n_rays, 3
    # acc_map: n_batch, n_rays

    def render_weights(alpha: torch.Tensor, eps=1e-8):
        # alpha: n_batch, n_rays, n_samples
        expanded_alpha = torch.cat([alpha.new_ones(*alpha.shape[:2], 1), 1. - alpha + eps], dim=-1)
        weights = alpha * torch.cumprod(expanded_alpha, dim=-1)[..., :-1]  # (n_batch, n_rays, n_samples)
        return weights

    if bg_image is not None:
        rgb[:, :, -1] = bg_image

    weights = render_weights(alpha, eps)  # (n_batch, n_rays, n_samples)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (n_batch, n_rays, 3)
    acc_map = torch.sum(weights, -1)  # (n_batch, n_rays)
    if bg_brightness < 0:  # smaller than zeros means we want to use noise as background
        bg_brightness = torch.rand_like(rgb_map)
    rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_brightness

    return weights, rgb_map, acc_map


def clamped_logit(x, high=1.0):
    x_high = torch.sigmoid(torch.tensor(high, dtyp=x.dtype, device=x.device))
    # scale = 1 - 2 * (1 - x_high)
    scale = 2 * x_high - 1
    x /= 2
    x -= 0.5
    x *= scale
    x += 0.5
    return torch.logit(x)


def shift_occ_to_msk(occ: torch.Tensor, min: float = 0.0, max: float = 1.0):
    # occ: (b, n, s), but indicates occupancy, [min, max], min outside, max inside
    msk = occ.max(dim=-1)[0]  # min_val, min_ind, find max along all samples
    msk -= min  # move
    msk /= max - min  # norm
    return msk


def apply_r(vds, se3):
    # se3: (B, N, 6), pts: (B, N, 3)
    B, N, _ = se3.shape
    se3 = se3.view(-1, se3.shape[-1])
    vds = vds.view(-1, vds.shape[-1])
    Rs = batch_rodrigues(se3[:, :3])  # get rotation matrix: (N, 3, 3)
    vds = torch.bmm(Rs, vds[:, :, None])[:, :, 0]  # batch matmul to apply rotation, and get (N, 3) back
    vds = vds.view(B, N, -1)
    return vds


def apply_rt(pts, se3):
    # se3: (B, N, 6), pts: (B, N, 3)
    B, N, _ = se3.shape
    se3 = se3.view(-1, se3.shape[-1])
    pts = pts.view(-1, pts.shape[-1])
    Rs = batch_rodrigues(se3[:, :3])  # get rotation matrix: (N, 3, 3)
    pts = torch.bmm(Rs, pts[:, :, None])[:, :, 0]  # batch matmul to apply rotation, and get (N, 3) back
    # TODO: retrain these...
    pts += se3[:, 3:]  # apply transformation
    pts = pts.view(B, N, -1)
    return pts


def expand_result_to_query_shape(func):
    def wrapper(*arg, **kwargs):
        query = arg[1]  # 0 is self, 1 is actually the first input
        val, ret = func(*arg, **kwargs)
        full = torch.zeros(*query.shape[:-1], val.shape[-1], device=val.device, dtype=val.dtype)
        full[ret.inds] = val
        return full
    return wrapper


def expand_result_to_query_shape_as_raw(func):
    def wrapper(*arg, **kwargs):
        query = arg[1]  # 0 is self, 1 is actually the first input
        val, ret = func(*arg, **kwargs)
        full = torch.zeros(*query.shape[:-1], val.shape[-1], device=val.device, dtype=val.dtype)
        full[ret.inds] = val
        ret.raw = full
        return ret
    return wrapper


def get_aspect_bounds(bounds) -> torch.Tensor:
    # bounds: B, 2, 3
    half_edge = (bounds[:, 1:] - bounds[:, :1]) / 2  # 1, 1, 3
    half_long_edge = half_edge.max(dim=-1, keepdim=True)[0].expand(-1, -1, 3)
    middle_point = half_edge + bounds[:, :1]  # 1, 1, 3
    return torch.cat([middle_point - half_long_edge, middle_point + half_long_edge], dim=-2)


@lru_cache
def get_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False) -> torch.Tensor:
    if preserve_aspect_ratio:
        bounds = get_aspect_bounds(bounds)
    n_batch = bounds.shape[0]

    # move to -1
    # scale to 1
    # scale * 2
    # move - 1

    move0 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    move0[:, :3, -1] = -bounds[:, :1]

    scale0 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    scale0[:, torch.arange(3), torch.arange(3)] = 1 / (bounds[:, 1:] - bounds[:, :1])

    scale1 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    scale1[:, torch.arange(3), torch.arange(3)] = 2

    move1 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    move1[:, :3, -1] = -1

    M = move1.matmul(scale1.matmul(scale0.matmul(move0)))

    return M  # only scale and translation has value


@lru_cache
def get_inv_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False) -> torch.Tensor:
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    invM = scale_trans_inverse(M)
    return invM


@lru_cache
def get_dir_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False) -> torch.Tensor:
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
    invM = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    return invM.mT


def scale_trans_inverse(M: torch.Tensor) -> torch.Tensor:
    n_batch = M.shape[0]
    invS = 1 / M[:, torch.arange(3), torch.arange(3)]
    invT = -M[:, :3, 3:] * invS[..., None]
    invM = torch.eye(4, device=M.device)[None].expand(n_batch, -1, -1)
    invM[:, torch.arange(3), torch.arange(3)] = invS
    invM[:, :3, 3:] = invT

    return invM


def affine_inverse(M: torch.Tensor) -> torch.Tensor:
    n_batch = M.shape[0]
    invR = M[:, :3, :3].mT
    invT = -invR.matmul(M[:, :3, 3:])
    invM = torch.eye(4, device=M.device)[None].expand(n_batch, -1, -1)
    invM[:, :3, :3] = invR
    invM[:, :3, 3:] = invT

    return invM


def ndc(pts, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    # both with batch dimension
    # pts has no last dimension
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def inv_ndc(pts, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    M = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def dir_ndc(dir, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    M = get_dir_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    dir = dir.matmul(R.mT)
    return dir


@lru_cache
def get_rigid_transform(poses: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor):
    from smplx.lbs import batch_rigid_transform, batch_rodrigues
    # pose: B, N, 3
    # joints: B, N, 3
    # parents: B, N
    # B, N, _ = poses.shape
    R = batch_rodrigues(poses.view(-1, 3))  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints, parents.view(-1))  # MARK: doc of this is wrong about parent
    return J, A


def get_rigid_transform_nobatch(poses: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor):
    from smplx.lbs import batch_rigid_transform, batch_rodrigues
    # pose: N, 3
    # joints: N, 3
    # parents: N
    R = batch_rodrigues(poses)  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints[None], parents)  # MARK: doc of this is wrong about parent
    J, A = J[0], A[0]  # remove batch dimension
    return J, A


def fast_sample_ray(ray_o, ray_d, near, far, img, msk, mask_at_box, nrays, split='train', body_ratio=0.5, face_ratio=0.0):
    msk = msk * mask_at_box
    if "train" in split:
        n_body = int(nrays * body_ratio)
        n_face = int(nrays * face_ratio)
        n_rays = nrays - n_body - n_face
        coord_body = torch.nonzero(msk == 1)
        coord_face = torch.nonzero(msk == 13)
        coord_rand = torch.nonzero(mask_at_box == 1)
        coord_body = coord_body[torch.randint(len(coord_body), [n_body, ])]
        coord_face = coord_face[torch.randint(len(coord_face), [n_face, ])]
        coord_rand = coord_rand[torch.randint(len(coord_rand), [n_rays, ])]
        coord = torch.cat([coord_body, coord_face, coord_rand], dim=0)
        mask_at_box = mask_at_box[coord[:, 0], coord[:, 1]]  # always True when training
    else:
        coord = torch.nonzero(mask_at_box == 1)
        # will not modify mask at box
    ray_o = ray_o[coord[:, 0], coord[:, 1]]
    ray_d = ray_d[coord[:, 0], coord[:, 1]]
    near = near[coord[:, 0], coord[:, 1]]
    far = far[coord[:, 0], coord[:, 1]]
    rgb = img[coord[:, 0], coord[:, 1]]
    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def take_jacobian(func: Callable, input: torch.Tensor, create_graph=False, vectorize=True, strategy='reverse-mode'):
    return torch.autograd.functional.jacobian(func, input, create_graph=create_graph, vectorize=vectorize, strategy=strategy)


def take_gradient(output: torch.Tensor,
                  input: torch.Tensor,
                  d_out: torch.Tensor = None,
                  create_graph: bool = True,
                  retain_graph: bool = True,
                  is_grads_batched: bool = False,
                  ):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,
                                is_grads_batched=is_grads_batched,
                                )
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


class MLP(GradModule):
    def __init__(self, input_ch=32, W=256, D=8, out_ch=257, skips=[4], actvn=nn.ReLU(), out_actvn=nn.Identity(), init=nn.Identity(), weight_norm=nn.Identity()):
        super(MLP, self).__init__()
        self.skips = skips
        self.linears = []
        for i in range(D + 1):
            I, O = W, W
            if i == 0:
                I = input_ch
            if i in skips:
                I = input_ch + W
            if i == D:
                O = out_ch
            self.linears.append(weight_norm(nn.Linear(I, O)))
        self.linears = nn.ModuleList(self.linears)
        self.actvn = actvn
        self.out_actvn = out_actvn

        for l in self.linears:
            init(l.weight)

    def forward(self, input: torch.Tensor):
        x = input
        for i, l in enumerate(self.linears):
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            if i == len(self.linears) - 1:
                a = self.out_actvn
            else:
                a = self.actvn
            x = a(l(x))  # actual forward
        return x


class SphereSignedDistanceField(GradModule):
    def __init__(self, d_in=63, d_hidden=256, n_layers=8, d_out=257, skips=[4], embedder=None):
        super(SphereSignedDistanceField, self).__init__()
        if embedder is not None:
            self.embedder = embedder

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        skips = [4]
        bias = 0.5
        scale = 1.0
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skips
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs * self.scale
        if hasattr(self, 'embedder'):
            inputs = self.embedder(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)


def project(xyz, K, RT):
    """
    xyz: [...N, 3], ... means some batch dim
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = xyz @ RT[:, :3].T + RT[:, 3:].T
    xyz = xyz @ K.T
    xy = xyz[..., :2] / xyz[..., 2:]
    return xy


def transform(xyz, RT):
    """
    xyz: [...N, 3], ... means some batch dim
    RT: [3, 4]
    """
    xyz = xyz @ RT[:, :3].T + RT[:, 3:].T
    return xyz


def fix_random(fix=True):
    if fix:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)


def load_model(net: nn.Module,
               optims: List[nn.Module],
               scheduler: nn.Module,
               recorder: nn.Module,
               model_dir,
               resume=True,
               strict=False,
               skips=[],
               only=[],
               allow_mismatch=[],
               epoch=-1,
               load_others=True):
    if not resume:
        log(f"removing trained weights: {model_dir}", 'red')
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth' and pth.endswith('.pth')
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch

    model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    log(f'loading model: {colored(model_path, "blue")}')
    pretrained_model = torch.load(model_path, 'cpu')

    pretrained_net = pretrained_model['net']
    if skips:
        keys = list(pretrained_net.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_net[k]

    if only:
        keys = list(pretrained_net.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_net[k]
    for key in allow_mismatch:
        if key in net.state_dict() and key in pretrained_net:
            net_parent = net
            pre_parent = pretrained_net
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                net_parent = getattr(net_parent, k)
                pre_parent = pre_parent[k]
            last_name = chain[-1]
            setattr(net_parent, last_name, nn.Parameter(pre_parent[last_name], requires_grad=getattr(net_parent, last_name).requires_grad))  # just replace without copying

    # for key in allow_mismatch:
    #     if key not in net.state_dict():
    #         continue
    #     net.state_dict()[key] = nn.Parameter(pretrained_model['net'][key], requires_grad=net.state_dict()[key].requires_grad)

    net.load_state_dict(pretrained_model['net'], strict=strict)
    log(f'loaded model at epoch: {colored(str(pretrained_model["epoch"]), "blue")}')
    if load_others:
        for i, optim in enumerate(optims):
            optim.load_state_dict(pretrained_model['optims'][i])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        recorder.load_state_dict(pretrained_model['recorder'])
        return pretrained_model['epoch'] + 1
    else:
        return 0


def save_model(net, optims, scheduler, recorder, model_dir, epoch, latest=False, rank=0, distributed=False):
    os.system('mkdir -p {}'.format(model_dir))
    if distributed:
        # all other ranks should consolidate the state dicts of the optimizer to the default rank: 0
        for opt in optims:
            if isinstance(opt, ZeroRedundancyOptimizer):
                opt.consolidate_state_dict()
                torch.cuda.synchronize()  # sync across devices to make sure that the state dict saved is full
        if rank != 0:
            return  # other processes don't need to save the model
    model = {
        'net': net.state_dict(),
        'optims': [optim.state_dict() for optim in optims],
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if latest:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 20:
        return
    os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def root_of_any(k, l):
    for s in l:
        a = accumulate(k.split('.'), lambda x, y: x + '.' + y)
        for r in a:
            if s == r:
                return True
    return False


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def load_network(
    net: nn.Module,
    model_dir,
    resume=True,
    epoch=-1,
    strict=False,
    skips=[],
    only=[],
    allow_mismatch=[],
):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        log(f'pretrained model: {model_dir} does not exist', 'red')
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth' and pth.endswith('.pth')
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    log(f'loading model: {colored(model_path, "blue")}')
    # ordered dict cannot be mutated while iterating
    # vanilla dict cannot change size while iterating
    pretrained_model = torch.load(model_path, 'cpu')
    pretrained_net = pretrained_model['net']

    if skips:
        keys = list(pretrained_net.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_net[k]

    if only:
        keys = list(pretrained_net.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_net[k]

    for key in allow_mismatch:
        if key in net.state_dict() and key in pretrained_net and not strict:
            net_parent = net
            pre_parent = pretrained_net
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                net_parent = getattr(net_parent, k)
                pre_parent = pre_parent[k]
            last_name = chain[-1]
            setattr(net_parent, last_name, nn.Parameter(pre_parent[last_name], requires_grad=getattr(net_parent, last_name).requires_grad))  # just replace without copying
    # for key in allow_mismatch:
    #     if key not in net.state_dict():
    #         continue
    #     net.state_dict()[key] = nn.Parameter(pretrained_net[key], requires_grad=net.state_dict()[key].requires_grad)

    net.load_state_dict(pretrained_net, strict=strict)
    log(f'loaded network at epoch: {colored(str(pretrained_model["epoch"]), "blue")}')
    return pretrained_model['epoch'] + 1


def logits_to_prob(logits):
    ''' Returns probabilities for logits
    Args:
        logits (tensor): logits
    '''
    odds = torch.exp(logits)
    probs = odds / (1 + odds)
    return probs


def prob_to_logits(probs, eps=1e-4):
    ''' Returns logits for probabilities.
    Args:
        probs (tensor): probability tensor
        eps (float): epsilon value for numerical stability
    '''
    probs = torch.clip(probs, a_min=eps, a_max=1 - eps)
    logits = torch.log(probs / (1 - probs))
    return logits


def get_bounds(xyz, padding=0.005):  # 5mm padding? really?
    # xyz: n_batch, n_points, 3

    min_xyz = torch.min(xyz, dim=1)[0]  # torch min with dim is ...
    max_xyz = torch.max(xyz, dim=1)[0]
    min_xyz -= padding
    max_xyz += padding
    bounds = torch.stack([min_xyz, max_xyz], dim=1)
    return bounds
    diagonal = bounds[..., 1:] - bounds[..., :1]  # n_batch, 1, 3
    bounds[..., 1:] = bounds[..., :1] + torch.ceil(diagonal / voxel_size) * voxel_size  # n_batch, 1, 3
    return bounds


def get_max_mem():
    return torch.cuda.max_memory_allocated() / 2 ** 20


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def normalize_sum(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.sum(dim=-1, keepdim=True) + eps)


def sigma_to_alpha(raw, dists=0.005, act_fn=F.softplus): return 1. - torch.exp(-act_fn(raw) * dists)


def sample_depth_near_far(near, far, n_samples: int, perturb: bool):
    # ray_o: n_batch, n_rays, 3
    # ray_d: n_batch, n_rays, 3
    # near: n_batch, n_rays
    # far: n_batch, n_rays
    # sample points and depth values for every ray between near-far
    # and possibly do some pertubation to introduce randomness in training
    # dists: n_batch, n_rays, n_steps
    # pts: n_batch, n_rays, n_steps, 3

    # calculate the steps for each ray
    t_vals = torch.linspace(0., 1., steps=n_samples, dtype=near.dtype, device=near.device)
    dists = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (dists[..., 1:] + dists[..., :-1])
        upper = torch.cat([mids, dists[..., -1:]], -1)
        lower = torch.cat([dists[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(*dists.shape, dtype=upper.dtype, device=upper.device)
        dists = lower + (upper - lower) * t_rand

    return dists


def sample_points_near_far(ray_o, ray_d, near, far, n_samples: int, perturb: bool):
    # ray_o: n_batch, n_rays, 3
    # ray_d: n_batch, n_rays, 3
    # near: n_batch, n_rays
    # far: n_batch, n_rays
    # sample points and depth values for every ray between near-far
    # and possibly do some pertubation to introduce randomness in training
    # dists: n_batch, n_rays, n_steps
    # pts: n_batch, n_rays, n_steps, 3

    # calculate the steps for each ray
    dists = sample_depth_near_far(near, far, n_samples, perturb)

    # (n_batch, n_rays, n_samples, 3)
    pts = ray_o[:, :, None] + ray_d[:, :, None] * dists[..., None]

    return pts, dists


def get_near_far_aabb(bounds: torch.Tensor, ray_o: torch.Tensor, ray_d: torch.Tensor, epsilon: float = 1e-8, return_raw=False):
    """
    calculate intersections with 3d bounding box
    bounds: n_batch, 2, 3, min corner and max corner
    ray_o: n_batch, n_points, 3
    ray_d: n_batch, n_points, 3, assume already normalized
    return: near, far (indexed by mask_at_box (bounding box mask))
    """
    if ray_o.ndim >= bounds.ndim:
        diff = ray_o.ndim - bounds.ndim
        for i in range(diff):
            bounds = bounds.unsqueeze(1)  # match the batch dimensions, starting from second

    # viewdir = ray_d / ray_d.norm(dim=-1, keepdim=True)
    # regularization for small values
    ray_d[(ray_d < epsilon) & (ray_d > -epsilon**2)] = epsilon
    ray_d[(ray_d > -epsilon**2) & (ray_d < epsilon)] = -epsilon
    # compute the intersection t on x, y, z plane for both bounding points of every ray
    # NOTE: here, min in tmin means the intersection with point bound_min, not minimum
    tmin = (bounds[..., :1, :] - ray_o) / ray_d  # (b, 1, 3) - (b, 1, 3) / (b, n, 3) -> (b, n, 3)
    tmax = (bounds[..., 1:, :] - ray_o) / ray_d  # (b, n, 3)
    # near plane is where the intersection has a smaller value on corresponding dimension than the other point
    t1 = torch.minimum(tmin, tmax)  # (b, n, 3)
    t2 = torch.maximum(tmin, tmax)
    # near plane is the maximum of x, y, z intersection point, entering AABB: enter every dimension
    near = t1.max(dim=-1)[0]  # (b, n)
    far = t2.min(dim=-1)[0]

    if return_raw:
        return near, far

    # box mask
    mask_at_box = near < far  # (b, n)
    # filter near far based on box
    near = near[mask_at_box]  # (b, m)
    far = far[mask_at_box]  # (b, m)
    return near, far, mask_at_box  # (b, n)


def get_bound_proposal(bounds: torch.Tensor, ray_o: torch.Tensor, ray_d: torch.Tensor, n_steps: int):
    near, far, mask_at_box = get_near_far_aabb(bounds, ray_o, ray_d)  # near, far already masked
    dists = sample_depth_near_far(near, far, n_steps, perturb=False)  # pts already masked
    return near, far, dists, mask_at_box


def compute_norm_diff(surf_pts: torch.Tensor, batch: dotdict[str, torch.Tensor], grad_decoder, diff_range: float, epsilon: float = 1e-6):
    n_batch, n_pts, D = surf_pts.shape
    surf_pts_neighbor = surf_pts + (torch.rand_like(surf_pts) - 0.5) * diff_range
    grad_pts = torch.cat([surf_pts, surf_pts_neighbor], dim=1)  # cat in n_masked dim
    grad: torch.Tensor = grad_decoder(grad_pts, batch)  # (n_batch, n_masked, 3)
    norm = grad / (grad.norm(dim=-1, keepdim=True) + epsilon)  # get normal direction
    norm_diff = (norm[:, n_pts:, :] - norm[:, :n_pts, :])  # neighbor - surface points

    return norm_diff, grad


def compute_val_pair_around_range(pts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], diff_range: float):
    # sample around input point and compute values
    # pts and its random neighbor are concatenated in second dimension
    # if needed, decoder should return multiple values together to save computation
    n_batch, n_pts, D = pts.shape
    neighbor = pts + (torch.rand_like(pts) - 0.5) * diff_range
    full_pts = torch.cat([pts, neighbor], dim=1)  # cat in n_masked dim
    raw: torch.Tensor = decoder(full_pts)  # (n_batch, n_masked, 3)
    return raw


def compute_diff_around_range(pts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], diff_range: float, norm_value: List[bool] = [True], dims: List[int] = [3]):
    # sample around input point and compute values, then take their difference
    # values are normalized based on settings (norm_value bool list)
    # pts and its random neighbor are concatenated in second dimension
    n_batch, n_pts, D = pts.shape
    neighbor = pts + (torch.rand_like(pts) - 0.5) * diff_range
    full_pts = torch.cat([pts, neighbor], dim=1)  # cat in n_masked dim
    raw: torch.Tensor = decoder(full_pts)  # (n_batch, n_masked, 3)
    diff = []
    for i, d in enumerate(dims):
        start = sum(dims[:i])
        stop = start + d
        part_value = raw[:, :, start:stop]
        if norm_value[i]:
            normed_value = normalize(part_value)
        part_diff = (normed_value[:, n_pts:, :] - normed_value[:, :n_pts, :])  # neighbor - surface points -> B, N, D
        diff.append(part_diff)
    diff = torch.cat(diff, dim=-1)

    return diff, raw
