import torch
import torch.nn as nn
from pytorch3d.ops import knn_points

from lib.config import cfg
from lib.networks import embedder
from lib.utils.base_utils import dotdict
from lib.utils.net_utils import GradModule, MLP, SphereSignedDistanceField, make_params, sdf_to_occ, batch_aware_indexing, multi_gather, multi_scatter, key_cache, normalize, alpha2raw, multi_gather_tris
from lib.utils.blend_utils import world_points_to_pose_points, world_dirs_to_pose_dirs, pose_points_to_tpose_points, affine_inverse, tpose_points_to_pose_points, pose_dirs_to_tpose_dirs, tpose_dirs_to_pose_dirs, blend_transform, torch_inverse_3x3, pose_dirs_to_world_dirs
from lib.utils.sample_utils import geodesic_knn
from typing import Tuple


class ResidualDeformation(GradModule):
    # unchanged expression (50) + full_pose (5 * 3) + translation as condition variables
    def __init__(self,
                 cond_dim=cfg.cond_dim,
                 multires=cfg.xyz_res,
                 resd_limit=cfg.resd_limit,
                 w_hidden=256,
                 n_hidden=8,
                 out_ch=3,
                 e_type='pe',
                 e_args={},
                 ):
        super(ResidualDeformation, self).__init__()
        self.resd_limit = resd_limit
        self.embedder, embed_dim = embedder.get_embedder(type=e_type, input_dims=3, multires=multires, **e_args)
        input_ch = embed_dim + cond_dim  # point query + conditional variable

        self.mlp = MLP(input_ch, w_hidden, n_hidden, out_ch, actvn=nn.ReLU())
        self.mlp.linears[-1].bias.data.zero_()

    def forward(self, points: torch.Tensor, cond: torch.Tensor):
        if len(cond.shape) == 2:
            cond = cond[:, None].expand(*points.shape[:2], -1)
        # points: B, N, 3
        points = self.embedder(points)
        input = torch.cat([points, cond], dim=-1)
        net = self.mlp(input)
        resd = torch.tanh(net) * self.resd_limit
        return resd


class SignedDistanceNetwork(GradModule):
    # UNISURF-like implementation
    # Official: https://github.com/autonomousvision/unisurf/blob/main/model/network.py
    # the official version looks an awful lot like VolSDF and IDR code
    # we decided not to use the bloated geometry init features
    def __init__(self,
                 multires=cfg.sdf_res,
                 feature_dim=cfg.feat_dim,
                 sdf_beta_init_value=cfg.sdf_beta_init_value,
                 finite_diff=cfg.sdf_finite_diff,
                 w_hidden=256,
                 n_hidden=8,
                 e_type='pe',
                 e_args={},
                 sdf_actvn=nn.Identity(),
                 ):  # defaults to no condition on geometry
        super(SignedDistanceNetwork, self).__init__()
        self.embedder, embed_dim = embedder.get_embedder(type=e_type, input_dims=3, multires=multires, **e_args)

        self._beta = make_params(torch.tensor(sdf_beta_init_value))  # optimizable logits multiplier

        input_ch = embed_dim  # point query + conditional variable
        out_ch = 1 + feature_dim  # occupancy + 256 features  # color

        self.mlp = SphereSignedDistanceField(input_ch, w_hidden, n_hidden, out_ch)
        self.sdf_occ_feat_grad = self.forward
        self.sdf_actvn = sdf_actvn
        self.finite_diff = finite_diff

    @property
    def beta(self):
        return self._beta.clamp(1e-9, 1e6)

    def sdf_feat(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: B, N, 3 + cond_dim
        # points already at ndc

        input = self.embedder(points)

        out = self.mlp(input)
        sdf, feat = out[..., :1], out[..., 1:]  # signed distance, features
        sdf = self.sdf_actvn(sdf)
        return sdf, feat

    def sdf_occ_feat(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points: B, N, 3 + cond_dim
        # points already at ndc
        sdf, feat = self.sdf_feat(points)
        occ = sdf_to_occ(sdf, self.beta)
        return sdf, occ, feat

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        return self.sdf_feat(points)[0]  # first output is the sigmoided occ value

    def feat(self, points: torch.Tensor) -> torch.Tensor:
        return self.sdf_feat(points)[1]  # second output is the network feature

    def occ(self, points: torch.Tensor) -> torch.Tensor:
        return self.sdf_occ_feat(points)[1]  # first output is the sigmoided occ value

    def grad(self, points: torch.Tensor) -> torch.Tensor:
        points.requires_grad_()
        with torch.enable_grad():
            sdf, feat = self.sdf_feat(points)
        return self.take_gradient(sdf, points)[..., :3]

    def take_gradient(self, sdf, points: torch.Tensor):
        if self.finite_diff > 0:
            px = points.detach().clone()
            py = points.detach().clone()
            pz = points.detach().clone()
            px[..., 0] += self.finite_diff
            py[..., 1] += self.finite_diff
            pz[..., 2] += self.finite_diff
            return torch.cat([self.sdf(px) - sdf, self.sdf(py) - sdf, self.sdf(pz) - sdf], dim=-1) / self.finite_diff
        else:
            return super(SignedDistanceNetwork, self).take_gradient(sdf, points)

    def forward(self, points: torch.Tensor):
        # points: B, N, 3
        points.requires_grad_()
        with torch.enable_grad():
            sdf, occ, feat = self.sdf_occ_feat(points)

        return sdf, occ, feat, self.take_gradient(sdf, points)[..., :3]  # will this be too slow? but it changes less


class RenderNetwork(GradModule):
    def __init__(self,
                 multires=cfg.view_res,
                 cond_dim=cfg.n_bones * 3,
                 feature_dim=cfg.feat_dim,
                 ):

        super(RenderNetwork, self).__init__()

        self.embedder, embed_dim = embedder.get_embedder(type='pe', input_dims=3, multires=multires)
        input_ch = 3 + feature_dim + embed_dim  # norm, feat, view
        out_ch = 3
        w_hidden = 256
        self.l0 = nn.utils.weight_norm(nn.Linear(input_ch, w_hidden))
        self.l1 = nn.utils.weight_norm(nn.Linear(w_hidden, w_hidden))
        self.l2 = nn.utils.weight_norm(nn.Linear(w_hidden, w_hidden))
        self.l3 = nn.utils.weight_norm(nn.Linear(w_hidden + cond_dim, w_hidden))
        self.l4 = nn.utils.weight_norm(nn.Linear(w_hidden, out_ch))
        self.actvn = nn.ReLU()

    def forward(self, view, grad, feat, cond):
        if len(cond.shape) == 2:
            cond = cond[: None].expand(*view.shape[:2], -1)
        # wpts: B, N, 3
        # view: B, N, 3
        # grad: B, N, 3
        # feat: B, N, 256
        # cond: B, N, 68
        view = self.embedder(view)
        input = torch.cat([view, grad, feat], dim=-1)
        net = input
        net = self.actvn(self.l0(net))
        net = self.actvn(self.l1(net))
        net = self.actvn(self.l2(net))
        net = torch.cat([net, cond], dim=-1)
        net = self.actvn(self.l3(net))
        net = self.l4(net)
        out = net
        rgb = torch.sigmoid(out)
        return rgb


class Network(GradModule):
    def __init__(self,
                 occ_th: float = cfg.occ_th,  # 0.5 as surface extraction point
                 dist_th: float = cfg.dist_th,  # how close to smpl to filter
                 surf_reg_th: float = cfg.surf_reg_th,  # 0.02 as surface proximity
                 blend_radius: float = cfg.blend_radius,
                 sample_vert_cnt: int = cfg.sample_vert_cnt,

                 cond_dim: int = cfg.cond_dim,
                 feature_dim: int = cfg.feat_dim,

                 lambertian: bool = cfg.lambertian,
                 use_geodesic_filter: bool = cfg.use_geodesic_filter,
                 ):
        super(Network, self).__init__()  # do not inherite network structure
        self.occ_th = occ_th
        self.dist_th = dist_th
        self.cond_dim = cond_dim
        self.lambertian = lambertian
        self.feature_dim = feature_dim
        self.surf_reg_th = surf_reg_th
        self.blend_radius = blend_radius
        self.sample_vert_cnt = sample_vert_cnt
        self.use_geodesic_filter = use_geodesic_filter

        # see https://github.com/pytorch/pytorch/issues/40769
        self.residual_deformation_network = ResidualDeformation(cond_dim=cond_dim)
        self.signed_distance_network = SignedDistanceNetwork()
        self.render_network = RenderNetwork(cond_dim=cond_dim)

    def network_geometry(self, x: torch.Tensor, batch):
        return alpha2raw(sdf_to_occ(self.inference_world_distance_field(x, batch), self.signed_distance_network.beta))

    def ray_marching(self, x: torch.Tensor, v: torch.Tensor, near: float = 0.05, far: float = 0.5, S: int = cfg.ray_samples, batch: dotdict = None):
        from lib.networks.relight.nerfactor_network import Network as NeRFactorNetwork
        return NeRFactorNetwork.ray_marching(self, x, v, near, far, S, batch)

    def geometry_visibility(self, x: torch.Tensor, n: torch.Tensor, batch: dotdict = None):
        from lib.networks.relight.nerfactor_network import Network as NeRFactorNetwork
        return NeRFactorNetwork.geometry_visibility(self, x, n, batch)

    def geometry_normal(self, x: torch.Tensor, batch: dotdict = None):
        from lib.networks.relight.nerfactor_network import Network as NeRFactorNetwork
        return NeRFactorNetwork.geometry_normal(self, x, batch)

    def residuals(self, bpts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        resd = self.residual_deformation_network(bpts, cond)
        return resd

    def residuals_and_observed_gradients(self, bpts: torch.Tensor, cond: torch.Tensor):
        # perform auto differentiation on input world points
        # points: B, N, 3
        bpts.requires_grad_()
        with torch.enable_grad():
            resd = self.residuals(bpts, cond)
            sdf = self.signed_distance_network.sdf(bpts + resd)
        ograd = self.take_gradient(sdf, bpts)
        return resd, ograd

    @staticmethod
    def condition_vector(batch):
        cond = batch.poses.view(batch.poses.shape[0], -1)  # B, 156
        return cond

    def world_to_bigpose(self, x: torch.Tensor, v: torch.Tensor, batch: dotdict[str, torch.Tensor], backward=False, transform=True, filtering=True, dist_th=None, bbox_margin=None):
        """
        transform: 
        Whether to apply the transformation to inputs points and directions
        If this argument is False, will query and blend transformaiton matrices of all points without filtering
        Will implicitly set filtering = False. Only the transform matrices will be returned in ret

        backward:
        Whether we're querying backward transformation matrices, if True, will implicitly set transform and filtering to False

        filtering:
        Whether to apply filtering, this will make the distance query hierarchical, by blending world space SMPL-H dists with canonical network queried distances
        """

        # Prepare the arguments
        if backward:
            transform = False

        if not transform:
            filtering = False

        if backward:
            # support transforming points from canonical to world space
            ppts = x
            space = 't'
        else:
            # transform points from world to smpl coords: remove global rotation and translation
            ppts = world_points_to_pose_points(x, batch.R, batch.Th)  # B, P, 3
            space = 'p'

        if filtering:
            dist = dist_th or self.dist_th
        else:
            dist = 1e9

        B, N, J = batch.weights.shape
        K = self.sample_vert_cnt

        sdf_batch, nn_batch, inds, S, d2, nn, ppts = geodesic_knn(ppts, batch[f'{space}verts'], batch[f'{space}norm'], batch.tverts, batch.tnorm, K, dist, use_geodesic_filter=self.use_geodesic_filter)  # always blend
        # sdf_batch: B, P, K
        # nn_batch: B, P, K
        # inds: B, S
        # S: scalar
        # d2: B, S, K
        # inds: B, S, K
        # ppts: B, S, 3

        # need to consider geodesic distance here -> canonical distance from the all K points to the closest one, if too large, remove these points with the closest points
        # filter points to get in the network with distance to smpl mesh
        bw = multi_gather(batch.weights[:, None].expand(B, S, N, J), nn)  # B, S, K, J
        w = (-d2 / (2 * self.blend_radius**2)).exp()  # B, S, K
        w = w / (w.sum(dim=-1, keepdim=True) + torch.finfo(w.dtype).eps)  # normalize bw weight
        bw = (w[..., None] * bw).sum(dim=-2)  # B, S, K, 1 * B, S, K, J -> B, S, J

        # precompute matrices for inverse linear blend skinning from smpl coords to canonical space
        big_A_bw = blend_transform(bw, batch.big_A)
        big_R_inv = torch_inverse_3x3(big_A_bw[..., :3, :3])
        A_bw = blend_transform(bw, batch.A)
        R_inv = torch_inverse_3x3(A_bw[..., :3, :3])

        if not transform:
            # only return the transformation matrices
            ret = dotdict()
            ret.A_bw = A_bw
            ret.big_A_bw = big_A_bw
            return ret

        # transform points from smpl coords to canonical space
        tpts = pose_points_to_tpose_points(ppts, A_bw=A_bw, R_inv=R_inv)  # tpose points
        bpts = tpose_points_to_pose_points(tpts, A_bw=big_A_bw, R_inv=big_R_inv)  # bigpose points

        ret = dotdict()
        ret.tpts = tpts
        ret.bpts = bpts
        ret.d2 = d2
        ret.nn = nn
        ret.inds = inds
        ret.nn_batch = nn_batch
        ret.sdf_batch = sdf_batch  # to match network return shape

        ret.A_bw = A_bw
        ret.R_inv = R_inv
        ret.big_A_bw = big_A_bw
        ret.big_R_inv = big_R_inv

        # deal with view direction if needed
        if v is not None:
            wvds = v
            pvds = world_dirs_to_pose_dirs(wvds, batch.R)  # B, P, 3
            pvds = multi_gather(pvds, inds)  # B, S, 3
            wvds = multi_gather(wvds, inds)  # B, S, 3
            tvds = pose_dirs_to_tpose_dirs(pvds, A_bw=A_bw, R_inv=R_inv)  # tpose points
            bvds = tpose_dirs_to_pose_dirs(tvds, A_bw=big_A_bw, R_inv=big_R_inv)  # bigpose points
            ret.wvds = wvds
            ret.pvds = pvds
            ret.tvds = tvds
            ret.bvds = bvds

        return ret

    def world_to_bigpose_transform(self, x: torch.Tensor, batch: dotdict[str, torch.Tensor], backward=False, **kwargs):
        # derived from world_to_bigpose
        ret = self.world_to_bigpose(x, None, batch, backward=backward, transform=False, **kwargs)
        A_bw = ret.A_bw
        big_A_bw = ret.big_A_bw
        B, P, _, _ = A_bw.shape

        # we need to compose R, Th, big_A_bw and A_bw
        # w2b: w2p, p2t, t2b
        padding = batch.R.new_zeros(B, 1, 4)  # B, 1, 4
        padding[..., -1] = 1.0
        p2w = torch.cat([batch.R, batch.Th[..., None]], dim=-1)  # B, 3, 4
        p2w = torch.cat([p2w, padding], dim=-2)  # B, 4, 4
        w2p = affine_inverse(p2w)  # B, 4, 4
        w2p = w2p[:, None].expand(B, P, 4, 4)  # B, P, 4, 4
        t2p = A_bw  # B, P, 4, 4
        p2t = affine_inverse(t2p)  # B, 4, 4
        t2b = big_A_bw  # B, P, 4, 4

        w2b = t2b @ p2t @ w2p
        return w2b

    def bigpose_to_world_transform(self, x: torch.Tensor, batch: dotdict[str, torch.Tensor], **kwargs):
        # derived from world_to_bigpose_transform
        w2b = self.world_to_bigpose_transform(x, batch, backward=True, **kwargs)
        return affine_inverse(w2b)

    def inference_world_geometry(self, x: torch.Tensor, batch: dotdict, smooth_transition=False, **kwargs) -> torch.Tensor:
        # derived from world_to_bigpose and inference_observed_distance_field
        # streamlined version of the forward function for only occupancy values
        ret = self.world_to_bigpose(x, None, batch, **kwargs)
        net_sdf = self.inference_observed_distance_field(ret.bpts, batch)

        dist_th = kwargs.get('dist_th', self.dist_th)

        # how do we make the distance value smoother?
        smpl_sdf = ret.sdf_batch.mean(dim=-1, keepdim=True)  # distance to 10 closest smplh vertices
        smpl_sdf = torch.where(smpl_sdf < -dist_th, smpl_sdf, smpl_sdf.abs())  # terminate at the actual geometry
        # this will bring the geometry to a more precise location when inside smpl, but outside the actual avatar
        if smooth_transition:
            d1 = multi_gather(smpl_sdf, ret.inds)  # smpl distance
            d2 = net_sdf  # queried neural sdf
            r = (d2.abs() / dist_th).clip(0, 1)  # alpha blending ratio
            net_sdf = d1 * r + d2 * (1 - r)  # a modified weighted average
        ret.sdf = multi_scatter(smpl_sdf, ret.inds, net_sdf)
        return ret

    def inference_world_distance_field(self, x: torch.Tensor, batch: dotdict, smooth_transition=False, **kwargs) -> torch.Tensor:
        ret = self.inference_world_geometry(x, batch, smooth_transition, **kwargs)
        return ret.sdf

    def inference_observed_geometry(self, x: torch.Tensor, batch: dotdict, smooth_transition=False, filtering=False, **kwargs) -> torch.Tensor:
        """
        For ablation study.
        Maybe query the SMPL-H vertices before querying the actual network

        Usually we don't need to query the canonical SMPL-H vertices since world_to_bigpose already does this
        Here the hierarchical query WILL NOT produce correct world space distance, but it will produce the correct canonical distance
        """

        ret = dotdict()

        dist_th = kwargs.get('dist_th', self.dist_th)

        if filtering:
            K = self.sample_vert_cnt
            sdf_batch, nn_batch, inds, S, d2, nn, x = geodesic_knn(x, batch.tverts, batch.tnorm, batch.tverts, batch.tnorm, K, dist_th)  # always filter
            ret.d2 = d2
            ret.nn = nn
            ret.inds = inds
            ret.nn_batch = nn_batch
            ret.sdf_batch = sdf_batch  # to match network return shape

        bpts = x
        cond = self.condition_vector(batch)[:, None].expand(*bpts.shape[:2], -1)  # condition variable, B, P, 156
        resd = self.residuals(bpts, cond)
        cpts = bpts + resd

        if cfg.smpl_distance:
            from bvh_distance_queries import BVH
            from pytorch3d.structures import Meshes
            bvh = BVH()

            d2, pts, fids, bcs = bvh(multi_gather_tris(batch.tverts, batch.faces), cpts)
            mesh = Meshes(batch.tverts, batch.faces)
            fnorm = mesh.faces_normals_padded()  # B, F, 3
            dist_batch = d2.sqrt()  # B, P, K
            dot_batch = ((cpts - pts) * multi_gather(fnorm, fids)).sum(dim=-1)  # B, P,
            sdf = dist_batch * dot_batch.sign()  # B, P, to match d2 return shape
            sdf = sdf[..., None]
        else:
            sdf = self.inference_canonical_distance_field(cpts)

        if filtering:
            # how do we make the distance value smoother?
            smpl_sdf = sdf_batch.mean(dim=-1, keepdim=True)  # distance to 10 closest smplh vertices
            smpl_sdf = torch.where(smpl_sdf < -dist_th, smpl_sdf, smpl_sdf.abs())  # terminate at the actual geometry
            # this will bring the geometry to a more precise location when inside smpl, but outside the actual avatar
            if smooth_transition:
                d1 = multi_gather(smpl_sdf, inds)  # smpl distance
                d2 = sdf  # queried neural sdf
                r = (d2.abs() / dist_th).clip(0, 1)  # alpha blending ratio
                sdf = d1 * r + d2 * (1 - r)  # a modified weighted average

            sdf = multi_scatter(smpl_sdf, inds, sdf)

        ret.sdf = sdf
        return ret

    def inference_observed_distance_field(self, x: torch.Tensor, batch: dotdict, smooth_transition=False, filtering=False, **kwargs) -> torch.Tensor:
        ret = self.inference_observed_geometry(x, batch, smooth_transition, filtering, **kwargs)
        return ret.sdf

    def inference_canonical_distance_field(self, x: torch.Tensor) -> torch.Tensor:
        cpts = x
        net_sdf = self.signed_distance_network.sdf(cpts)
        return net_sdf

    def forward_geometry(self, x: torch.Tensor, v: torch.Tensor, d: torch.Tensor, batch: dotdict, **kwargs):
        # warp from world space to canonical space (uncorrected)
        out = self.world_to_bigpose(x, v, batch, **kwargs)
        bpts = out.bpts

        # pass throught the network modules (backward residual deformation + signed distance network + shape-from-shading renderer)
        cond = self.condition_vector(batch)[:, None].expand(*bpts.shape[:2], -1)  # condition variable, B, P, 156
        bpts.requires_grad_()
        with torch.enable_grad():
            resd = self.residuals(bpts, cond)  # a residual deformation for non-rigid deformation, prone to overfitting in monocular settings
            cpts = bpts + resd
            sdf, occ, feat = self.signed_distance_network.sdf_occ_feat(cpts)  # canonical points
        ograd = self.take_gradient(sdf, bpts)  # observed gradients also considering the loss

        # differences from anisdf's implementation
        norm = normalize(ograd)  # bigpose normal, should not use world space normal since the renderer should live in canonical bigpose space
        norm = pose_dirs_to_tpose_dirs(norm, A_bw=out.big_A_bw, R_inv=out.big_R_inv)  # tpose normal
        norm = tpose_dirs_to_pose_dirs(norm, A_bw=out.A_bw, R_inv=out.R_inv)  # pose space normal
        norm = pose_dirs_to_world_dirs(norm, batch.R)  # world normal direction?
        norm = normalize(norm)  # bigpose normal, should not use world space normal since the renderer should live in canonical bigpose space

        ret = dotdict()  # remove unwanted geometry output
        if self.training:  # geometry regularization
            ret.residuals = resd
            ret.observed_gradients = ograd
            ret.gradients = self.take_gradient(sdf, cpts)  # intrinsic sdf gradients

        if not self.training:  # for visualization
            out.cpts = cpts
            out.bpts = bpts
            out.resd = resd

        out.cpts = cpts
        out.norm = norm
        out.feat = feat
        out.cond = cond
        out.occ = occ
        out.sdf = sdf
        return ret, out

    def forward(self, x: torch.Tensor, v: torch.Tensor, d: torch.Tensor, batch: dotdict, **kwargs):
        # forward geometry
        ret, out = self.forward_geometry(x, v, d, batch, **kwargs)

        # forward renderer
        if not self.training:  # maybe fix rendering material for testing?
            if cfg.fix_material >= 0 or cfg.always_fix_material:  # using the material of the first frame for better relighting results
                out.cond = batch.train_motion.poses[:, cfg.fix_material].view(batch.train_motion.poses[:, 0].shape[0], -1)[:, None].expand(*out.bpts.shape[:2], -1)  # condition variable, B, P, 156 (manuel may not work here)
        rgb = self.render_network(out.bvds, out.norm, out.feat, out.cond)  # use canonical norm?

        # apply back the previously done filtering
        raw = torch.cat([out.norm, rgb, out.occ], dim=-1)
        if not self.training:
            raw = torch.cat([out.cpts, out.bpts, out.resd, raw], dim=-1)
        raw = multi_scatter(raw.new_zeros(*x.shape[:-1], raw.shape[-1]), out.inds, raw)

        # prepare output parameters
        ret.raw = raw

        return ret
