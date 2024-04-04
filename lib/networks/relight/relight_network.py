import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.data_utils import load_mesh
from lib.utils.blend_utils import pose_dirs_to_tpose_dirs, pose_dirs_to_world_dirs, tpose_dirs_to_pose_dirs
from lib.utils.net_utils import normalize, MLP, load_network, freeze_module, make_params, make_buffer, take_gradient, multi_scatter, batch_aware_indexing, multi_gather, unfreeze_module, multi_scatter_, GradModule, torch_dot
from lib.utils.relight_utils import Microfacet, gen_light_xyz
from lib.networks.deform import base_network
from lib.networks.embedder import get_embedder
from lib.utils.log_utils import log


class Network(base_network.Network):
    def __init__(
        self,
        fresnel_f0=cfg.fresnel_f0,
        geometry_mesh=cfg.geometry_mesh,
        geometry_pretrain=cfg.geometry_pretrain,
        xyz_res=cfg.relight_xyz_res,
        view_res=cfg.relight_view_res,  # are there still view encoding needs?
        env_h=cfg.env_h,
        env_w=cfg.env_w,
        env_r=cfg.env_r,
        achro_light=cfg.achro_light,
        xyz_noise_std=cfg.xyz_noise_std,
        *args,
        **kwargs,
    ):
        super(Network, self).__init__(*args, **kwargs)  # do not inherite network structure

        # load the pretrained model of the network
        load_network(self, geometry_pretrain, strict=False)
        freeze_module(self.render_network)

        self.xyz_embedder, self.xyz_dim = get_embedder(xyz_res, 3)  # no parameters
        self.view_embedder, self.view_dim = get_embedder(view_res, 3)  # no parameters

        self.prepare_relight_network()
        self.prepare_relight_metadata()

    def prepare_relight_network(self):
        self.albedo_network = MLP(input_ch=self.feature_dim, W=cfg.relight_network_width, D=cfg.relight_network_depth, out_ch=3, actvn=nn.Softplus(beta=100), out_actvn=lambda x: cfg.albedo_slope * torch.sigmoid(x) + cfg.albedo_bias, init=nn.init.kaiming_normal_)  # albedo and roughness can only be in [0, 1]
        self.roughness_network = MLP(input_ch=self.feature_dim, W=cfg.relight_network_width, D=cfg.relight_network_depth, out_ch=1, actvn=nn.Softplus(beta=100), out_actvn=lambda x: cfg.roughness_slope * torch.sigmoid(x) + cfg.roughness_bias, init=nn.init.kaiming_normal_)  # albedo and roughness can only be in [0, 1]

    def prepare_relight_metadata(self,
                                 fresnel_f0=cfg.fresnel_f0,
                                 env_h=cfg.env_h,
                                 env_w=cfg.env_w,
                                 env_r=cfg.env_r,
                                 achro_light=cfg.achro_light,
                                 xyz_noise_std=cfg.xyz_noise_std,
                                 lambert_only=cfg.lambert_only,
                                 glossy_only=cfg.glossy_only,
                                 ):

        self.microfacet = Microfacet(f0=fresnel_f0, lambert_only=lambert_only, glossy_only=glossy_only)  # no parameters

        # use an optimizable fixed environment lighting system for now
        if achro_light:
            self.global_env_map_ = make_params(torch.rand(env_h * cfg.envmap_upscale, env_w * cfg.envmap_upscale, 1) * cfg.envmap_init_intensity)
        else:
            self.global_env_map_ = make_params(torch.rand(env_h * cfg.envmap_upscale, env_w * cfg.envmap_upscale, 3) * cfg.envmap_init_intensity)

        xyz, area = gen_light_xyz(env_h, env_w, env_r, device='cpu')  # eH, eW, 3; eH, eW
        sharp = 1 / (area / torch.pi).sqrt()  # as in tangent, H, W, how much was obsecured
        self.light_xyz_ = make_buffer(xyz)
        self.light_area = make_buffer(area)
        self.light_sharp = make_buffer(sharp)

        # other regularization related configuration entry
        self.xyz_noise_std = xyz_noise_std
        self.achro_light = achro_light
        self.env_h, self.env_w, self.env_r = env_h, env_w, env_r

    @property
    def light_xyz(self):
        if self.training:
            return self.light_xyz_ + torch.randn_like(self.light_xyz_) * cfg.light_xyz_noise_std
        else:
            return self.light_xyz_

    @property
    def global_env_map(self):
        # TODO: fix this ugly impl
        return F.softplus(self.global_env_map_.expand(*self.global_env_map_.shape[:2], 3))

    def forward(self, x: torch.Tensor, v: torch.Tensor, d: torch.Tensor, batch: dotdict):  # NOTE: viewdirection is not used here
        ret, out = self.forward_geometry(x, None, d, batch)

        # first we need to get the albedo and specular parameters from the MLP
        # ebd = self.xyz_embedder(out.cpts)  # albedo lives in canonical
        # input = torch.cat([ebd, out.feat], dim=-1)
        albedo = self.albedo_network(out.feat)
        roughness = self.roughness_network(out.feat)

        # apply back the previously done filtering
        raw = torch.cat([albedo, roughness, out.norm, out.occ], dim=-1)
        if not self.training:
            raw = torch.cat([out.cpts, out.bpts, out.resd, raw], dim=-1)
        ret.raw = multi_scatter(raw.new_zeros(*x.shape[:-1], raw.shape[-1]), out.inds, raw)

        # apply regularization
        if self.training:
            # define smoothness loss on jitter points
            cpts: torch.Tensor = out.cpts  # B, P, 3
            xyz_noise = torch.normal(mean=0, std=self.xyz_noise_std, size=cpts.shape, device=cpts.device)
            xyz_jitter = cpts + xyz_noise
            # ebd_jitter = self.xyz_embedder(xyz_jitter)  # albedo lives in canonical
            feat_jitter = self.signed_distance_network.feat(xyz_jitter)
            # input = torch.cat([ebd_jitter, feat_jitter], dim=-1)
            ret.albedo = albedo
            ret.roughness = roughness
            ret.albedo_jitter = self.albedo_network(feat_jitter)
            ret.roughness_jitter = self.roughness_network(feat_jitter)

        return ret
