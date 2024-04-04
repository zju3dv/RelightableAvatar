import os
import cv2
import torch
import numpy as np
from lib.config import cfg
from lib.utils.data_utils import project, read_mask_by_img_path
from lib.utils.base_utils import dotdict

from . import pose_dataset


class Dataset(pose_dataset.Dataset):
    def __init__(self, data_root, human, ann_file, split, **kwargs):
        super(Dataset, self).__init__(data_root, human, ann_file, split)

    def load_image_size(self):
        pass

    def load_view(self):
        self.view = [0, ]
        self.num_cams = 1  # only one camera for mesh extraction

    def load_camera(self):
        pass

    def get_indices(self, index):
        latent_index = index
        i = latent_index
        if latent_index == -1:
            i = 0  # load data of first frame if index is -1
        frame_index = self.i + i * self.i_intv  # recompute frame index for i
        return latent_index, frame_index, -1, -1

    def __getitem__(self, index):  # TODO: might get -1 in optimization, but doesn't matter now
        latent_index, frame_index, _, _ = self.get_indices(index)
        i = frame_index

        # load SMPL & pose & human related parameters
        ret = self.get_blend(i)

        voxel_size = cfg.voxel_size
        if cfg.vis_can_mesh or cfg.vis_tpose_mesh:
            bounds = ret.tbounds
        else:
            bounds = ret.wbounds
        x = torch.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
        y = torch.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
        z = torch.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
        pts = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).float().numpy()

        meta = {
            'pts': pts,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': 0,  # no view
        }
        ret.update(meta)
        ret.meta.update(meta)

        return dotdict(ret)
