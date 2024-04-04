import torch
import numpy as np
from torch import nn
from math import prod
from sympy import nextprime
from typing import Tuple, Union

from lib.utils.base_utils import dotdict
from lib.utils.net_utils import make_buffer, make_params


class PositionalEncoding(nn.Module):
    def __init__(self, multires=10, periodic_fns=[torch.sin, torch.cos], retain_input=True):
        super(PositionalEncoding, self).__init__()
        freq_bands = 2.**torch.linspace(0., multires-1, steps=multires)  # (multires)
        freq_bands = freq_bands[..., None, None].expand(multires, len(periodic_fns), 1).clone()  # (multires, 2, 1)
        self.freq_bands = make_buffer(freq_bands)
        # self.register_buffer('freq_bands', freq_bands)
        self.multires = multires
        self.periodic_fns = periodic_fns
        self.retain_input = retain_input

    def get_dim(self, dim):
        return self.freq_bands.numel() * dim + (dim if self.retain_input else 0)

    def forward(self, inputs: torch.Tensor):
        # inputs: B, N, 3
        n_b_dim = len(inputs.shape)-1
        dim = inputs.shape[-1]
        ori_inputs = inputs
        inputs = inputs.view(*inputs.shape[:-1], 1, 1, inputs.shape[-1])  # (B, N, 1, 1, 3)
        inputs = inputs * self.freq_bands[(None,)*n_b_dim]  # (B, N, 1, 1, 3) * (1, 1, multires, 2, 3) -> (B, N, multires, 2, 3)
        inputs = torch.cat([self.periodic_fns[i](t) for i, t in enumerate(torch.split(inputs, 1, dim=-2))], dim=-2)
        inputs = inputs.view(*ori_inputs.shape[:-1], self.freq_bands.numel() * dim)  # (B, N, embed_dim - 3?)
        if self.retain_input:
            inputs = torch.cat([ori_inputs, inputs], dim=-1)
        return inputs


class HashEncoding(nn.Module):
    def __init__(self,
                 bbox=np.array([
                     [-2, -2, -2],
                     [2,  2,  2]
                 ]),
                 n_levels=16,
                 n_features_per_level=16,
                 b=1.38,
                 log2_hashmap_size=20,
                 base_resolution=16,
                 sum=True,
                 sum_over_features=True,
                 separate_dense=True,
                 include_input=True,  # this will pass gradient better to input, but if you're using uvt, no need
                 ps=[1, 19349663, 83492791],
                 ):
        """
        WIP:
        best iter speed: separate_dense = True
        best performace: separate_dense = False, sum_over_features = True

        TODO: add configure entry for number of dimension
        """
        super().__init__()
        self.t = log2_hashmap_size
        self.n_levels = n_levels
        self.include_input = include_input
        self.n_entries_per_level = nextprime(2**log2_hashmap_size)
        self.ps = ps

        self.b = b
        self.f = n_features_per_level
        self.base_resolution = base_resolution

        self.bounds = make_buffer(torch.tensor(np.array(bbox).reshape((2, 3))).float())

        # every level should have this number of entries per side
        # we'd like the border to be mapped inside 0, 1
        self.entries_num = [int((self.base_resolution * self.b**i)) for i in range(self.n_levels)]
        self.entries_cnt = [self.entries_num[i] ** 3 for i in range(self.n_levels)]
        self.entries_size = [1 / (self.entries_num[i] - 1) for i in range(self.n_levels)]
        self.entries_min = [0 for i in range(self.n_levels)]

        self.entries_size = make_buffer(torch.tensor(self.entries_size))
        self.entries_num = make_buffer(torch.tensor(self.entries_num))
        self.entries_min = make_buffer(torch.tensor(self.entries_min))
        self.entries_cnt = make_buffer(torch.tensor(self.entries_cnt))
        self.entries_sum = make_buffer(self.entries_cnt.cumsum(dim=-1))

        self.start_hash = self.n_levels
        for i in range(n_levels):
            if self.entries_cnt[i] > self.n_entries_per_level:
                self.start_hash = i
                break
        self.len_hash = self.n_levels - self.start_hash
        self.separate_dense = separate_dense and self.start_hash  # when everything needs to be hashed for example when body using using small table
        if self.separate_dense:
            data = torch.zeros((self.n_levels, self.n_entries_per_level, self.f))
            nn.init.kaiming_normal_(data)  # NOTE: initialization matters! separate_dense doesn't work well if we initialize the self.dense and self.hash data separately
            dense = torch.cat([data[i, :self.entries_cnt[i], :] for i in range(self.start_hash)], dim=0)
            hash = data[self.start_hash:, :, :]
            self.dense = make_params(dense)  # sum(non-hash), F
            self.hash = make_params(hash)  # H, T, F
        else:
            self.hash = make_params(torch.zeros((self.n_levels, self.n_entries_per_level, self.f)))  # H, T, F
            nn.init.kaiming_normal_(self.hash)

        self.offsets = make_buffer(torch.tensor([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.],
        ]).float())

        self.sum = sum
        self.sum_over_features = sum_over_features

        self.out_dim = 0

        if self.sum:
            if self.sum_over_features:
                self.out_dim += self.n_levels
            else:
                self.out_dim += self.f
        else:
            self.out_dim += self.f * self.n_levels

        if include_input:
            self.out_dim += 3

    def forward(self, xyz: torch.Tensor):
        bash = xyz.shape  # batch shape
        xyz = xyz.view(prod(bash[:-1]), xyz.shape[-1])

        N, _ = xyz.shape  # N, 3
        xyz = (xyz - self.bounds[0]) / (self.bounds[1] - self.bounds[0])  # normalized, N, 3

        ind_xyz = xyz[None].expand(self.n_levels, -1, -1)  # L, N, 3
        flt_xyz = ind_xyz / self.entries_size[:, None, None]  # L, N, 3
        int_xyz = (flt_xyz[:, :, None] + self.offsets[None, None]).long()  # will round to zero, L, N, 8, 3
        int_xyz = int_xyz.clip(self.entries_min[:, None, None, None], self.entries_num[:, None, None, None]-1)
        off_xyz = flt_xyz - int_xyz[:, :, 0]  # L, N, 3

        sh = self.start_hash
        nl = self.n_levels

        # x as first digit, y as second digit, z as last digit -> S, N, 8
        ind_dense: torch.Tensor = \
            int_xyz[:sh, ..., 0] * (self.entries_num[:sh])[:, None, None] ** 2 + \
            int_xyz[:sh, ..., 1] * (self.entries_num[:sh])[:, None, None] + \
            int_xyz[:sh, ..., 2]
        if self.separate_dense:
            ind_dense[1:] = ind_dense[1:] + self.entries_sum[:self.start_hash-1][:, None, None]  # S, N, 8

        # hashing -> H, N, 8
        ind_hash: torch.Tensor = (
            int_xyz[sh:, ..., 0] * self.ps[0] ^
            int_xyz[sh:, ..., 1] * self.ps[1] ^
            int_xyz[sh:, ..., 2] * self.ps[2]
        ) % self.n_entries_per_level
        if not self.separate_dense:
            ind = torch.cat([ind_dense, ind_hash], dim=0)

        # data: L, T, F, ind: L, N, 8 -> L, N, 8, F feature
        # NOTE: gather backward is much faster than index_select
        # val = self.data[torch.arange(nl, dtype=torch.long, device=ind.device)[..., None, None], ind, :]  # -> L, N, 8, F
        L, T, F = self.n_levels, self.n_entries_per_level, self.f
        S, H = self.start_hash, self.n_levels - self.start_hash

        if self.separate_dense:
            val_dense = self.dense.gather(dim=0, index=ind_dense.view(S * N * 8)[..., None].expand(-1, F)).view(S, N, 8, F)
            val_hash = self.hash.gather(dim=1, index=ind_hash.view(H, N * 8)[..., None].expand(-1, -1, F)).view(H, N, 8, F)
            val = torch.cat([val_dense, val_hash], dim=0)
        else:
            val = self.hash.gather(dim=1, index=ind.view(L, N * 8)[..., None].expand(-1, -1, F)).view(L, N, 8, F)

        # off: L, N, 3, sets: 8, 3 -> L, N, :, 3 and :, :, 8, 3, compute xyz distance to the other corner, mul: multiplier
        mul_xyz = (1 - self.offsets[None, None]) + (2 * self.offsets[None, None] - 1.) * off_xyz[:, :, None]
        mul_xyz = mul_xyz[..., 0] * mul_xyz[..., 1]  # L, N, 8
        val = (mul_xyz[..., None] * val).sum(dim=-2)  # trilinear interpolated feature, L, N, F

        # feature aggregation
        val = val.permute(1, 0, 2)  # N, L, F
        if self.sum:
            if self.sum_over_features:
                val = val.sum(dim=-1)  # N, F, NOTE: sum over features seems to be producing better results...
            else:
                val = val.sum(dim=-2)  # N, L, NOTE: sum over features seems to be producing better results...
        else:
            val = val.reshape(-1, L*F)  # N, L*F

        # feature boosting
        if self.include_input:
            val = torch.cat([xyz, val], dim=-1)

        val = val.view(*bash[:-1], val.shape[-1])
        return val

    def extra_repr(self) -> str:
        # will be visible in print
        self.extra_dict = dotdict()
        self.extra_dict.base_resolution = self.base_resolution
        self.extra_dict.n_levels = self.n_levels
        # self.extra_dict.include_input = self.include_input
        self.extra_dict.t = self.t
        # self.extra_dict.ps = self.ps
        self.extra_dict.b = self.b
        self.extra_dict.f = self.f

        return ', '.join([k + '=' + str(v) for k, v in self.extra_dict.items()])


def get_embedder(multires=10, input_dims=3, type='pe', *args, **kwargs) -> Tuple[Union[PositionalEncoding, HashEncoding, int]]:
    if type == 'pe':
        embedder = PositionalEncoding(multires=multires, *args, **kwargs)
        embedder.out_dim = embedder.get_dim(input_dims)
        return embedder, embedder.out_dim
    else:
        embedder = HashEncoding(*args, **kwargs)
        return embedder, embedder.out_dim
