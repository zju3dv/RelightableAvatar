# This file is reused when we're performing textured rendering or just plain old rendering
import os

from lib.config import cfg
from lib.config.config import Output
from lib.utils.base_utils import dotdict
from lib.utils.log_utils import log
from lib.utils.data_utils import save_image
from . import base_visualizer


class Visualizer(base_visualizer.Visualizer):
    def prepare_result_paths(self):
        data_dir = f'data/novel_view/{cfg.exp_name}'
        motion_name = os.path.splitext(os.path.basename(cfg.test_motion))[0]

        if cfg.perform:
            img_path = f'{data_dir}/{motion_name}/{{type}}/frame{{view:04d}}_view{{frame:04d}}{cfg.vis_ext}'
        elif 'sfm' in cfg.test_dataset_module or 'mipnerf360' in cfg.test_dataset_module:  # special treatment for sfm datasets
            img_path = f'{data_dir}/{{type}}/frame{{frame:04d}}_view{{view:04d}}{cfg.vis_ext}' # TODO: this is evil
        else:
            img_path = f'{data_dir}/frame_{{frame:04d}}/{{type}}/{{view:04d}}{cfg.vis_ext}'

        self.result_dir = os.path.dirname(img_path)
        self.img_path = img_path

    def visualize_single_type(self, output: dotdict, batch: dotdict, type: Output=Output.Rendering):
        img_pred = Visualizer.generate_image(output, batch, type)
        frame_index = batch.meta.frame_index.item()
        view_index = batch.meta.view_index.item()
        self.view_index = view_index  # for generating video
        self.frame_index = frame_index  # for generating video

        img_path = self.img_path.format(type=type.name.lower(), frame=frame_index, view=view_index)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        save_image(img_path, img_pred)
