import os
import torch
import trimesh
import numpy as np

from copy import deepcopy

from lib.config import cfg
from lib.utils.log_utils import log
from lib.utils.loss_utils import l2
from lib.utils.net_utils import inv_ndc, ndc
from lib.utils.base_utils import dotdict
from lib.networks.deform import base_network
from lib.networks.renderer import mesh_renderer

# NOTE: largesteps package itself contains memory leak issue on its own
# in its custom torch.autograd.Function, there's reference to the solver
# which holds som cupy memories (like workspace memory of Cholesky factorization)
# see this open issue on torch https://github.com/pytorch/pytorch/issues/7343
# And even if we replace the largesteps implementation with a no cyclic reference one
# which can be verified by adding weakref callbacks (print deallocation messages)
# there still exists a small amount of memory leak when calling scsrsm2_analysis
# from cupy_backends.cuda.libs.cusparse, which is cloes-sourced by Nvidia...
# so we choose just to ignore it since it's rare that you should be optimizing
# a lot of different meshes in one run of the program, this is worrying
# NOTE: update: found a solution for the cuSPARSE memory leak: https://github.com/rgl-epfl/large-steps-pytorch/pull/7
# and tested that the solvers was deleted even if we do not remove the ctx.solver in backward
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform


class Renderer(mesh_renderer.Renderer):
    def __init__(self, net: base_network.Network):
        super(Renderer, self).__init__(net)
        # TODO: is there a better way to impl this different run pass?
        # cannot simply call render_master here since it needs CUDA
        self.master = None

        self.net.train()  # some double auto-differentiation queries the network training state
        self.net.requires_grad_(False)  # do not update the network itself

    def render(self, batch):
        log(f'optimizing:  {batch.meta.latent_index.item()}')
        if not self.master:
            return self.render_master(batch)  # will fill self.master and self.prev

        # TODO: figure out how to update the blend weight when doing this optimization
        vertices = self.master.vertices  # get the tpose (or not) vertice location
        vertices: torch.Tensor = torch.tensor(vertices, device=batch.meta.latent_index.device, dtype=torch.float32)[None]

        if cfg.use_ndc:
            vertices = ndc(vertices, batch.tbounds)

        # Eventaully used Adam for 200 iteration step, loss around 1e-8 - 1e-9
        cond = self.net.get_cond(vertices, batch, category='tpts')  # TODO: this API might change
        if self.prev is None:
            occ, ograd = self.net.oocc_ograd(vertices, cond)  # TODO: this API might change
            resd = -(occ - 0.5) / ograd  # if bigger than 0.5, let's say the offset is 0.1 then 0.1 / grad means the residual to go up 0.1
            closest = vertices + resd
            root = -(self.net.resd(closest, cond) + closest - vertices) + closest
            root: torch.Tensor = root.contiguous()  # initial guess
        else:
            root = self.prev.detach().clone()
        vertices: torch.Tensor = vertices.detach()  # remove from graph

        root = to_differential(self.M, root[0])[None]
        root.requires_grad_()
        optimizer = AdamUniform([root], cfg.root_iter_step_size)  # TODO: can be faster by continuously optimization

        for ith_iter in range(1, cfg.root_iter_max+1):
            with torch.enable_grad():  # might have disabled gradients on a global level
                # Get cartesian coordinates for parameterization
                query = from_differential(self.M, root[0], 'Cholesky')[None]
                resd = self.net.resd(query, cond)  # pass through model
                cpts = query + resd
                cycle_loss = l2(cpts, vertices)  # compute cycle consistency loss
                loss = cycle_loss

            optimizer.zero_grad()  # clean gradients
            loss.backward()  # update gradients: will do third diff on killing
            optimizer.step()  # perform update

            if ith_iter >= cfg.root_iter_min and loss.item() < cfg.root_iter_target:
                break

        log(f'iteration:   {ith_iter}')
        log(f'loss:        {loss.item()}')

        root = query
        if cfg.use_ndc:
            root = inv_ndc(root, batch.tbounds)
        self.prev = root  # remember previous location
        vertices = root.detach().cpu().numpy()[0]
        ret = deepcopy(self.master)
        ret.vertices = vertices
        ret.mesh.vertices = vertices
        return ret

    def render_master(self, batch):
        if cfg.resume_tracking_at >= 0:
            log(f'resuming:    {cfg.resume_tracking_at}')
            batch.meta.latent_index[0] = cfg.resume_tracking_at
            track_path = os.path.join(cfg.anim_dir, 'track_mesh', f'{cfg.resume_tracking_at:04d}.npz')
            track = np.load(track_path)
            track = dotdict({**track})
            tverts = torch.tensor(track.vertices[None], device=batch.meta.latent_index.device, dtype=torch.float32)
            faces = torch.tensor(track.triangles[None], device=tverts.device, dtype=torch.long)

            master_path = os.path.join(cfg.anim_dir, f'can_mesh.npz')
            ret = np.load(master_path)
            ret = dotdict({**ret})  # the real master
            ret.mesh = trimesh.Trimesh(ret.vertices, ret.triangles)  # FIXME: resume tracking is not working properly
        else:
            # will do a regular tpose mesh computation for frame 0
            batch.meta.latent_index[0] = 0
            ret = super().render(batch)
            batch.meta.latent_index[0] = -1
            # normally this is batch.meta.latent_index: 0
            # so we do another deformation to get the actual vertices
            tverts = torch.tensor(ret.vertices[None], device=batch.meta.latent_index.device, dtype=torch.float32)
            faces = torch.tensor(ret.triangles[None], device=tverts.device, dtype=torch.long)
            cond = self.net.get_cond(tverts, batch, category='tpts')  # TODO: this API might change
            if cfg.use_ndc:
                tverts = ndc(tverts, batch.tbounds)
            cverts = self.net.tpose_pts_to_can_pts(tverts, cond).cpts
            if cfg.use_ndc:
                cverts = inv_ndc(cverts, batch.tbounds)
            ret.vertices = cverts[0].detach().cpu().numpy()

        self.prev = tverts
        self.faces = faces
        self.master = ret
        # NOTE: when restarting the vertices are not in canonical space
        # but vertice location doesn't really matter in this laplacian implementation D-A (degree - adjacency)
        # https://www.geneseo.edu/~aguilar/public/notes/Graph-Theory-HTML/ch4-laplacian-matrices.html
        # NOTE: no batch dim in these matrices
        self.M = compute_matrix(tverts[0], faces[0], cfg.root_iter_lambda)  # laplacian
        # while the coalescing process will accumulate the multi-valued elements into a single value using summation:
        # https://pytorch.org/docs/stable/sparse.html
        return self.master
