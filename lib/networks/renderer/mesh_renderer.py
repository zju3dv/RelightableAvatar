import torch
import mcubes
import trimesh
import numpy as np
from torch import nn

from lib.config import cfg
from lib.utils.log_utils import log
from lib.utils.sample_utils import sample_closest_points_on_surface, sample_blend_K_closest_points
from lib.utils.base_utils import dotdict
from lib.utils.data_utils import alpha2sdf
from lib.networks.deform import base_network

from pytorch3d.ops import knn_points


class Renderer(nn.Module):
    def __init__(self, net: base_network.Network):
        super().__init__()
        self.net = net

    def batchify_rays(self, wpts, occ_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = occ_decoder(wpts[i:i + chunk])
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def render(self, batch):
        pts = batch.pts  # B, x, y, z, 3
        shape = pts.shape  # B, x, y, z, 3

        if cfg.vis_can_mesh or cfg.vis_tpose_mesh:
            vert_key = 'tverts'
        elif cfg.vis_posed_mesh:
            vert_key = 'wverts'

        pts = pts.view(1, -1, 3)
        log('filtering')
        tnorm = sample_blend_K_closest_points(pts, batch[vert_key], K=cfg.sample_vert_cnt)  # preserve memory
        tnorm = tnorm[..., -1]
        inside = tnorm < cfg.dist_th
        pts = pts[inside]  # filter points

        if cfg.vis_can_mesh or (cfg.vis_tpose_mesh and batch.meta.latent_index == -1):
            def sdf_decoder(x): return -self.net.signed_distance_network.sdf(x[None])[0, ..., 0]

        elif cfg.vis_posed_mesh:
            def sdf_decoder(x): return -self.net.inference_world_geometry(x[None], batch).sdf[0, ..., 0]

        elif cfg.vis_tpose_mesh:
            cond = self.net.condition_vector(batch)

            def sdf_decoder(x):
                resd = self.net.residual_deformation_network(x[None], cond)
                return -self.net.signed_distance_network.sdf(x[None] + resd)[0, ..., 0]
        else:
            raise NotImplementedError()

        log('inferencing')
        occ = self.batchify_rays(pts, sdf_decoder, self.net, cfg.network_chunk_size, batch)

        # create marching cubes oval
        cube = np.ones(shape[1:-1]).reshape(-1) * -10
        inside = inside.detach().cpu().numpy()[0]
        cube[inside] = occ
        cube = cube.reshape(shape[1:-1])
        cube = np.pad(cube, 10, mode='constant', constant_values=-10)

        if 'signed_distance_network' in dir(self.net) and cfg.mesh_th_to_sdf:
            mesh_th = alpha2sdf(cfg.mesh_th, self.net.signed_distance_network.beta.detach().cpu().numpy())
        else:
            mesh_th = cfg.mesh_th

        log('marching cubes')
        verts, faces = mcubes.marching_cubes(cube, mesh_th)
        verts = (verts - 10) * cfg.voxel_size[0]
        if cfg.vis_can_mesh or cfg.vis_tpose_mesh:
            bounds = batch.tbounds
        elif cfg.vis_posed_mesh:
            bounds = batch.wbounds

        verts = verts + bounds[0, 0].detach().cpu().numpy()

        # * MESH SIMPLIFICATION FOR COMPACT REPRESENTATION
        log('simplifying mesh')
        # From v and f, generate trimesh
        mesh = trimesh.Trimesh(verts, faces)
        mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.vertices))  # get largest component (removing artifacts automatically)

        if cfg.mesh_simp_face > 0:
            mesh = mesh.simplify_quadratic_decimation(cfg.mesh_simp_face)

        verts = mesh.vertices.astype(np.float32)  # verts and faces might get updated during unwrapping
        faces = mesh.faces.astype(np.int32)  # c++ volrend needs this to be int32

        if hasattr(self.net, 'albedo_network'):
            if cfg.vis_can_mesh or (cfg.vis_tpose_mesh and batch.meta.latent_index == -1):
                def material_decoder(x):
                    sdf, occ, feat = self.net.signed_distance_network.sdf_occ_feat(x[None])
                    albedo = self.net.albedo_network(feat)[0]
                    roughness = self.net.albedo_network(feat)[0]
                    return albedo, roughness

            elif cfg.vis_posed_mesh:
                def material_decoder(x):
                    out = self.net.inference_world_geometry(x[None], batch)
                    albedo = self.net.albedo_network(out.feat)[0]
                    roughness = self.net.albedo_network(out.feat)[0]
                    return albedo, roughness

            elif cfg.vis_tpose_mesh:
                cond = self.net.condition_vector(batch)

                def material_decoder(x):
                    resd = self.net.residual_deformation_network(x[None], cond)
                    sdf, occ, feat = self.net.signed_distance_network.sdf_occ_feat(x[None] + resd)
                    albedo = self.net.albedo_network(feat)[0]
                    roughness = self.net.albedo_network(feat)[0]
                    return albedo, roughness
            else:
                raise NotImplementedError()
            log('extracting albedo and roughness')
            albedo, roughness = material_decoder(torch.as_tensor(verts, device='cuda'))

        # * EXTRA INFORMATION: BLEND WEIGHTS
        log('extracting blend weights')
        verts_tensor = torch.from_numpy(verts).cuda(non_blocking=True)[None]
        if cfg.surface_blend_weight:
            weights, dists = sample_closest_points_on_surface(verts_tensor, batch[vert_key], batch.faces, batch.weights)  # B, N, D & B, N, 1
        else:
            weights, dists = sample_blend_K_closest_points(verts_tensor, batch[vert_key], batch.weights, K=cfg.sample_vert_cnt)
        weights = weights[0].detach().cpu().numpy()

        if cfg.vis_can_mesh or cfg.vis_tpose_mesh:  # joints coorresponding the current setup
            joints = batch.big_joints[0].detach().cpu().numpy()
        elif cfg.vis_posed_mesh:
            joints = batch.joints[0].detach().cpu().numpy()
        tjoints = batch.tjoints[0].detach().cpu().numpy()
        parents = batch.parents[0].detach().cpu().numpy().astype(np.int32)  # c++ volrend needs this to be int32

        # TODO: PERFORM A GOOD UV UNWRAPPING

        ret = dotdict()
        ret.joints = joints
        ret.tjoints = tjoints
        ret.parents = parents
        ret.weights = weights
        ret.verts = verts
        ret.faces = faces
        if 'albedo' in locals(): ret.albedo = albedo
        if 'roughness' in locals(): ret.roughness = roughness

        log(f'statistics: verts: {len(verts)}, faces: {len(faces)}')
        return ret
