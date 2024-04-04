import os
import cv2
import torch
import imageio
import numpy as np
import torch.utils.data as data

from typing import List
from os.path import join
from termcolor import colored
from os import system, listdir

from lib.config import cfg
from lib.utils.log_utils import log, run
from lib.utils.base_utils import dotdict
from lib.utils.sem_utils import palette_to_index, palette_to_onehot
from lib.utils.relight_utils import linear2srgb, read_hdr, area_hot_img
from lib.utils.data_utils import get_rigid_transform, sample_ray, load_image, load_unchanged, read_mask_by_img_path, batch_rodrigues, get_bounds, to_tensor, to_numpy, load_dotdict, load_unchanged_image

from pytorch3d.structures import Meshes

from easymocap.config.baseconfig import load_object, Config
from easymocap.bodymodel.smplx import SMPLHModel, SMPLModel

# Naming convenstion for dataset functions:
# 1. load_stuff is to load some shared info from the disk (like the pose of bigpose or smpl face / weights like meta-data)
# 2. get_stuff is to prepare for per-index batch input data for the training of this index (implemented in a nested structure for now to make it DRY)


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()
        # do not change the order of data loading (has dependent structures)
        self.load_meta(data_root, human, ann_file, split)  # load shared metadata across dataset
        self.load_view()  # determine view to use for the current mode
        self.load_gt()  # for prefetching, for now, just skeleton
        self.load_ims_inds()
        self.load_ims_data()  # only load the data, no strange processing
        self.load_smpl()  # load motion, body model and smpl related stuff
        self.load_bigpose()
        self.load_lighting()
        self.load_image_size()

    def load_image_size(self):
        # reduce the image resolution by ratio
        img_post = self.ims[0] if hasattr(self, 'ims') else self.annots['ims'][0]['ims'][0]  # will this lead to error?
        if isinstance(img_post, List): img_post = img_post[0]
        img_path = os.path.join(self.data_root, img_post)
        img = imageio.imread(img_path)
        H, W = img.shape[:2]
        self.H, self.W = H, W

    def load_meta(self, data_root, human, ann_file, split):
        self.data_root = data_root
        self.human = human
        self.split = split

        self.annots = np.load(join(data_root, ann_file), allow_pickle=True).item()
        self.cams = self.annots['cams']
        self.bkgd_root = join(self.data_root, cfg.bkgd)
        self.nrays = cfg.n_rays

        if os.path.exists(self.bkgd_root):
            try:
                self.BGs = np.stack([load_image(join(self.bkgd_root, f'{cam:02d}.jpg')) for cam in range(len(self.cams['K']))])
            except:
                pass

    def load_view(self):
        num_cams = len(self.cams['K'])
        training_view = cfg.training_view if len(cfg.training_view) else list(range(num_cams))
        if len(cfg.test_view) == 0:
            test_view = [i for i in range(num_cams)]  # use all of the views for testing
        else:
            test_view = cfg.test_view  # use user specified test view
        view = training_view if 'train' in self.split else test_view
        view = [v for v in view if v < num_cams]
        self.view = view
        self.num_cams = len(self.view)

    def load_ims_inds(self):
        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame if 'train' in self.split else cfg.num_eval_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame

        # for pose indexing
        self.i = i
        self.ni = ni
        self.i_intv = i_intv

        # assert self.i + (self.ni - 1) * self.i_intv < len(self.motion.poses), 'motion out of range'

    def load_ims_data(self):
        # for training images & training indexing
        # for pose indexing
        i = self.i
        ni = self.ni
        i_intv = self.i_intv

        # data for training requires gt for rendering
        # TODO: duplicated code here
        num_cams = len(self.annots['ims'][0]['ims'])
        training_view = cfg.training_view if len(cfg.training_view) else list(range(num_cams))
        if len(cfg.test_view) == 0:
            test_view = [i for i in range(num_cams)]  # use all of the views for testing
        else:
            test_view = cfg.test_view  # use user specified test view
        view = training_view if 'train' in self.split else test_view
        view = [v for v in view if v < num_cams]
        self.view = view
        self.num_cams = len(self.view)

        self.ims = np.array([
            np.array(ims_data['ims'])[self.view]
            for idx, ims_data in enumerate(self.annots['ims'][i:i + ni * i_intv][::i_intv]) if idx * i_intv + i not in cfg.skip
        ])
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.view]
            for idx, ims_data in enumerate(self.annots['ims'][i:i + ni * i_intv][::i_intv]) if idx * i_intv + i not in cfg.skip
        ])
        self.ims = self.ims.ravel()
        self.cam_inds = self.cam_inds.ravel()

    def load_gt(self):
        pass

    def load_olat(self):
        # MARK: unused for now
        H, W = cfg.env_h, cfg.env_w
        # olats = range(H * W)
        olats = cfg.olats
        olat_inten, ambient_inten = cfg.olat_inten, cfg.ambient_inten
        # (1) OLAT
        novel_olats = dotdict()
        for idx in olats:
            i, j = idx // W, idx % W
            name = f'olat{i:04d}-{j:04d}'
            if cfg.test_light and name not in cfg.test_light and name != cfg.replace_light: continue
            one_hot = area_hot_img(H, W, 3, i, j)
            probe = olat_inten * one_hot + ambient_inten
            novel_olats[name] = dotdict()
            novel_olats[name].probe = probe
            if cfg.vis_ground_shading and cfg.ground_attach_envmap:
                novel_olats[name].image = cv2.resize(probe, (cfg.env_image_w, cfg.env_image_h), interpolation=cv2.INTER_LINEAR)  # upscale
        self.novel_olats = novel_olats

        from lib.utils.data_utils import save_image
        for n, i in novel_olats.items():
            save_image(join(cfg.lighting_dir, '16x32', n + '.hdr'), i.probe)

    def load_probe(self):
        # Novel lighting conditions for relighting at test time:
        # (2) Light probes
        novel_probes = dotdict()
        for path in sorted(os.listdir(join(cfg.lighting_dir, '16x32'))):
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]  # without extension
            if cfg.test_light and name not in cfg.test_light and name != cfg.replace_light: continue
            novel_probes[name] = dotdict()
            novel_probes[name].probe = read_hdr(join(cfg.lighting_dir, '16x32', path))
            if cfg.vis_ground_shading and cfg.ground_attach_envmap:  # avoid loading large files
                hdr_path = join(cfg.lighting_dir, '8k', path)
                if os.path.exists(hdr_path):
                    novel_probes[name].image = read_hdr(hdr_path)
                else:
                    novel_probes[name].image = novel_probes[name].probe
        self.novel_probes = novel_probes

    def load_lighting(self):
        self.load_olat()
        self.load_probe()
        self.novel_lights = {**self.novel_probes, **self.novel_olats}

        # (4) Make lights stronger
        for k, v in self.novel_lights.items():  # copy by reference
            v.probe = v.probe * cfg.light_multiplier
            if cfg.vis_ground_shading and cfg.ground_attach_envmap:  # avoid loading large files
                v.image = v.image * cfg.light_multiplier

        # Logging
        if cfg.replace_light:
            log(f'replacing learned environment map with {colored(cfg.replace_light, "blue")} for rendering')

    def load_smpl(self):
        self.train_motion = load_dotdict(join(self.data_root, cfg.train_motion))
        self.test_motion = load_dotdict(join(self.data_root, cfg.test_motion))
        if self.split == 'train':
            self.motion = self.train_motion
        else:
            self.motion = self.test_motion
        self.shapes = self.train_motion.shapes[0]  # assume shape are shared

        # Load tpose stuff from geometry prior
        if cfg.use_geometry:
            geometry_mesh = load_dotdict(cfg.geometry_mesh)
            self.parents = geometry_mesh.parents.astype(np.int64)
            self.weights = geometry_mesh.weights.astype(np.float32)

            self.faces = geometry_mesh.faces.astype(np.int64)
            self.tverts = geometry_mesh.verts.astype(np.float32)
            self.tjoints = geometry_mesh.tjoints.astype(np.float32)
        # Load tpose stuff from bodymodel
        else:
            cfg_model = Config.load(join(self.data_root, cfg.body_model))  # whatever for now
            cfg_model.module = cfg_model.module.replace('SMPLHModelEmbedding', 'SMPLHModel')
            cfg_model.args.device = 'cpu'
            bodymodel: SMPLModel = load_object(cfg_model.module, cfg_model.args)
            self.bodymodel = bodymodel
            self.parents = to_numpy(bodymodel.parents).astype(np.int64)
            self.weights = to_numpy(bodymodel.weights).astype(np.float32)
            tpose_poses = torch.zeros_like(torch.from_numpy(self.motion.poses[:1]))
            tpose_shapes = torch.from_numpy(self.shapes[None])
            self.faces = to_numpy(bodymodel.faces_tensor).astype(np.int64)
            self.tverts = to_numpy(bodymodel(poses=tpose_poses, shapes=tpose_shapes)[0])
            self.tjoints = to_numpy(bodymodel(poses=tpose_poses, shapes=tpose_shapes, return_smpl_joints=True, return_verts=False)[0])

        self.tbounds = get_bounds(self.tverts)

    def load_bigpose(self):
        big_poses = np.zeros([len(self.tjoints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_J, big_A = get_rigid_transform(big_poses, self.tjoints, self.parents)  # n_bones, 4, 4
        inv_big_A = np.linalg.inv(big_A)
        self.big_poses, self.big_joints, self.big_A, self.inv_big_A = big_poses, big_J, big_A, inv_big_A

        # Load bigpose stuff from geometry prior
        if cfg.use_geometry:
            pass
        # Load bigpose stuff from bodymodel
        else:
            bpose_poses = torch.from_numpy(self.big_poses).view(-1)[None]
            bpose_shapes = torch.from_numpy(self.shapes[None])
            self.tverts = to_numpy(self.bodymodel(poses=bpose_poses, shapes=bpose_shapes)[0])  # update tpose vertices to bigpose

        self.tbounds = get_bounds(self.tverts)

    def get_normal(self, index):
        img_path = self.ims[index]
        norm_path = join(self.data_root, img_path.replace('images', 'normal'))[:-4] + '.png'
        if not os.path.exists(norm_path):
            norm_path = join(self.data_root, img_path.replace('images', 'normal'))[:-4] + '.jpg'
        norm = load_image(norm_path)
        norm = 2 * (norm - 0.5)
        return norm

    def get_semantic(self, index):
        # load schp colored coded semantic map
        img_path = self.ims[index]
        sem_path = join(self.data_root, img_path.replace('images', 'schp'))[:-4] + '.png'
        if not os.path.exists(sem_path):
            sem_path = join(self.data_root, img_path.replace('images', 'schp'))[:-4] + '.jpg'
        sem = load_unchanged(sem_path)
        sem = palette_to_onehot(sem)
        return sem

    def get_mask(self, index):
        msk = read_mask_by_img_path(self.data_root, self.ims[index], cfg.erode_dilate_mask, cfg.mask)
        H, W = int(msk.shape[0] * cfg.ratio), int(msk.shape[1] * cfg.ratio)  # mark: maybe mismatch with self.H, self.W
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        # load camera parameters
        view_index = self.cam_inds[index]
        K = np.array(self.cams['K'][view_index], dtype=np.float32)
        D = np.array(self.cams['D'][view_index], dtype=np.float32)

        # update camera parameters based on scaling
        K[:2] = K[:2] * cfg.ratio

        # undistort image & mask -> might overload CPU
        msk = cv2.undistort(msk, K, D)
        return msk

    def get_image_and_mask(self, index):
        img_path = join(self.data_root, self.ims[index])
        img = load_image(img_path)  # why do we need a separated get_mask function?
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)  # mark: maybe mismatch with self.H, self.W
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        # load camera parameters
        view_index = self.cam_inds[index]
        K = np.array(self.cams['K'][view_index], dtype=np.float32)
        D = np.array(self.cams['D'][view_index], dtype=np.float32)

        # update camera parameters based on scaling
        K[:2] = K[:2] * cfg.ratio

        # undistort image & mask -> might overload CPU
        img = cv2.undistort(img, K, D)

        # masking images
        msk = self.get_mask(index)

        if cfg.mask_bkgd:
            msk = torch.from_numpy(msk)
            img = torch.from_numpy(img)
            img[msk == 0] = 0
            img = img.numpy()
            msk = msk.numpy()

        return img, msk

    def get_lbs_params(self, i):
        poses = self.motion.poses[i].reshape(-1, 3)
        Rh = self.motion.Rh[i]
        Th = self.motion.Th[i]  # MARK: Th requires an empty point dimension
        shapes = self.train_motion.shapes[0] # always use training shape

        poses_tensor = torch.from_numpy(poses).view(-1)[None]
        shapes_tensor = torch.from_numpy(shapes).view(-1)[None]
        Rh_tensor = torch.from_numpy(Rh).view(-1)[None]
        Th_tensor = torch.from_numpy(Th).view(-1)[None]

        J, A = get_rigid_transform(poses, self.tjoints, self.parents)  # n_bones, 4, 4
        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)

        # Load posed stuff from bodymodel
        if cfg.use_geometry:
            from lib.utils.data_utils import as_torch_func, as_numpy_func
            from lib.utils.blend_utils import pose_points_to_tpose_points, tpose_points_to_pose_points, pose_points_to_world_points
            txyz = as_numpy_func(pose_points_to_tpose_points)(self.tverts, self.weights, self.big_A)
            pxyz = as_numpy_func(tpose_points_to_pose_points)(txyz, self.weights, A)
            wxyz = as_numpy_func(pose_points_to_world_points)(pxyz, R, Th)
        else:
            wxyz = to_numpy(self.bodymodel(poses=poses_tensor, shapes=shapes_tensor, Rh=Rh_tensor, Th=Th_tensor)[0])

            pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        return wxyz, pxyz, A, J, Rh, Th, poses, shapes

    def get_blend(self, i):
        wverts, pverts, A, J, Rh, Th, poses, shapes = self.get_lbs_params(i)
        pbounds = get_bounds(pverts)
        wbounds = get_bounds(wverts)

        ret = dotdict()
        ret.meta = dotdict()

        Rs = batch_rodrigues(poses)  # N, 3, 3

        # blend weight
        meta = {
            'A': A,
            'Rs': Rs,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': self.tbounds,
            'shapes': shapes,
            'poses': poses,
            'joints': J,
            'tjoints': self.tjoints,
            'big_joints': self.big_joints,
            'parents': self.parents,
            'big_poses': self.big_poses,
            'big_A': self.big_A,
            # excessive storage?
            'motion': self.motion,  # all motion data stored here
            'train_motion': self.train_motion,  # all motion data stored here
            'test_motion': self.test_motion,  # all motion data stored here
        }
        ret.update(meta)
        # vert sampling
        meta = {
            'wverts': wverts,
            'pverts': pverts,
            'tverts': self.tverts,
            'faces': self.faces,
            'weights': self.weights
        }
        ret.update(meta)

        wverts, pverts, self.tverts = to_tensor([wverts, pverts, self.tverts])
        faces = to_tensor(self.faces)
        mesh = Meshes([wverts, pverts, self.tverts], [faces, faces, faces])
        wnorm, pnorm, tnorm = to_numpy(mesh.verts_normals_list())
        meta = {
            'wnorm': wnorm,
            'pnorm': pnorm,
            'tnorm': tnorm,
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R': R, 'Rh': Rh, 'Th': Th}
        ret.update(meta)

        # Novel lighting condition (relighting temporal solution)
        ret.novel_lights = self.novel_lights

        return ret

    def get_indices(self, index):
        # store index data
        latent_index = index // len(self.view)

        # find frame index
        i = int(os.path.basename(self.ims[index])[:-4])
        frame_index = i  # make sure no out of bound sampling for training

        # load camera parameters
        view_index = self.cam_inds[index]  # should always be zero

        # store camera sub
        cam_index = view_index

        return latent_index, frame_index, view_index, cam_index

    def get_gt(self, index):
        # read images & masks
        img, msk = self.get_image_and_mask(index)

        # get meta indices
        latent_index, frame_index, view_index, cam_index = self.get_indices(index)

        # load camera parameters
        K = np.array(self.cams['K'][cam_index], dtype=np.float32)
        D = np.array(self.cams['D'][cam_index], dtype=np.float32)
        R = np.array(self.cams['R'][cam_index], dtype=np.float32)
        T = np.array(self.cams['T'][cam_index], dtype=np.float32) / 1000.

        # update camera parameters based on scaling
        H, W = img.shape[:2]
        K[:2] = K[:2] * cfg.ratio

        # load SMPL & pose & human related parameters
        ret = self.get_blend(frame_index)

        # store image parameter
        meta = {
            'img': img,
            'msk': msk,
        }
        ret.update(meta)

        # store camera parameters
        meta = {
            'cam_K': K,
            'cam_R': R,
            'cam_T': T,
            'cam_RT': np.concatenate([R, T], axis=1),
            'H': H,
            'W': W,
        }
        ret.update(meta)
        ret.meta.update(meta)  # keep a copy on the cpu

        # store camera background images
        if hasattr(self, 'BGs'):
            BG = np.array(self.BGs[view_index], dtype=np.float32)
            BG = cv2.resize(BG, (W, H), interpolation=cv2.INTER_AREA)
            BG = cv2.undistort(BG, K, D)
            meta = {
                "cam_BG": BG,
            }
            ret.update(meta)
            ret.meta.update(meta)

        meta = {
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': view_index,
        }
        ret.update(meta)
        ret.meta.update(meta)
        return ret

    def __getitem__(self, index):
        ret = self.get_gt(index)
        img, msk, H, W, K, R, T, RT = ret.img, ret.msk, ret.H, ret.W, ret.cam_K, ret.cam_R, ret.cam_T, ret.cam_RT
        wbounds = ret.wbounds  # need for near far and mask at box

        # sample rays
        rgb, ray_o, ray_d, near, far, coord, mask_at_box = sample_ray(
            img, msk, K, R, T, wbounds, cfg.n_rays, self.split, cfg.subpixel_sample,
            cfg.body_sample_ratio, cfg.face_sample_ratio)

        # compute occupancy (mask value), whether sampled point is in mask
        msk = msk[coord[:, 0], coord[:, 1]].astype(np.float32)

        # store ray data
        meta = {
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'coord': coord,
            'msk': msk,
            'mask_at_box': mask_at_box,
        }
        ret.update(meta)

        return ret

    def __len__(self):
        # this will make sure no out of bound sampling when training
        return len(self.ims)
