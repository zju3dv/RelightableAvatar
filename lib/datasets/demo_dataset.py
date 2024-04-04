import os
import imageio
import numpy as np

from lib.config import cfg
from lib.utils.log_utils import log
from lib.utils.render_utils import gen_path
from lib.utils.data_utils import get_rays_within_bounds
from lib.datasets import pose_dataset


class Dataset(pose_dataset.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__(data_root, human, ann_file, split)
        self.load_render()

    def load_render(self):
        self.render_w2c = gen_path(self.RT, cfg.novel_view_center, cfg.novel_view_z_off)
        self.num_cams = len(self.render_w2c)
        self.K = self.Ks[0]
        self.K[0, 0] *= cfg.novel_view_ixt_ratio
        self.K[1, 1] *= cfg.novel_view_ixt_ratio
        if len(self.render_w2c) > len(self.motion.poses) and cfg.perform:
            log(f'will render {len(self.render_w2c)} views on only {len(self.motion.poses)} poses', 'yellow')

    def get_indices(self, index):
        if cfg.perform:  # ? BUG
            latent_index = index
        else:
            latent_index = 0
        view_index = index
        frame_index = self.i + latent_index * self.i_intv  # recompute frame index for i
        cam_index = view_index
        return latent_index, frame_index, view_index, cam_index

    def __getitem__(self, index):
        latent_index, frame_index, view_index, cam_index = self.get_indices(index)
        ret = self.get_blend(frame_index)

        # reduce the image resolution by ratio
        if cfg.H <= 0 or cfg.W <= 0:
            img_path = os.path.join(self.data_root, self.annots['ims'][0]['ims'][0])  # need to get H, W
            img = imageio.imread(img_path)
            H, W = img.shape[:2]
            H, W = int(H * cfg.ratio), int(W * cfg.ratio)
            K = self.K
        else:
            H, W = cfg.H, cfg.W
            K = np.zeros((3, 3), dtype=np.float32)
            K[2, 2] = 1
            K[0, 0] = H * cfg.novel_view_ixt_ratio
            K[1, 1] = H * cfg.novel_view_ixt_ratio
            K[0, 2] = H / 2
            K[1, 2] = H / 2

        RT = self.render_w2c[view_index]
        R, T = RT[:3, :3], RT[:3, 3:]
        ray_o, ray_d, near, far, mask_at_box = get_rays_within_bounds(H, W, K, R, T, ret.wbounds)

        # store camera parameters
        meta = {
            "cam_K": K,
            "cam_R": R,
            "cam_T": T,
            "cam_RT": np.concatenate([R, T], axis=1),
            'H': H,
            'W': W,
            'RT': self.RT,
            'Ks': self.Ks,
        }
        ret.update(meta)
        ret.meta.update(meta)

        # store ray data
        meta = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
        }
        ret.update(meta)

        # store index data
        meta = {
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': view_index,
        }
        ret.update(meta)
        ret.meta.update(meta)
        return ret

    def __len__(self):
        return len(self.render_w2c)
