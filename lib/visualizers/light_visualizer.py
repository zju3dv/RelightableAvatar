# This file is reused when we're performing textured rendering or just plain old rendering
import os
import torch
from os.path import join

from lib.config import cfg
from lib.config.config import Output
from lib.utils.base_utils import dotdict
from lib.utils.data_utils import save_image, to_cuda
from lib.utils.parallel_utils import parallel_execution
from . import base_visualizer


class Visualizer(base_visualizer.Visualizer):
    def prepare_result_paths(self):
        data_dir = f'data/novel_light/{cfg.exp_name}'
        data_dir = join(data_dir, cfg.extra_prefix) if cfg.extra_prefix else data_dir  # differentiate between video and evals
        motion_name = os.path.splitext(os.path.basename(cfg.test_motion))[0]

        if len(cfg.test_view) == 1:
            img_path = f'{data_dir}/view_{{view:04d}}/{{type}}/{{frame:04d}}{cfg.vis_ext}'
        elif cfg.num_eval_frame == 1:
            img_path = f'{data_dir}/frame{{frame:04d}}/{{type}}/{{view:04d}}{cfg.vis_ext}'
        else:
            img_path = f'{data_dir}/{motion_name}/{{type}}/frame{{frame:04d}}_view{{view:04d}}{cfg.vis_ext}'

        self.result_dir = os.path.dirname(img_path)
        self.img_path = img_path

    def visualize_single_type(self, output: dotdict, batch: dotdict, type: Output = Output.Rendering):
        frame_index = batch.meta.frame_index.item()
        view_index = batch.meta.view_index.item()
        self.view_index = view_index  # for generating video
        self.frame_index = frame_index  # for generating video

        img_path = self.img_path.format(type=type.name.lower(), frame=frame_index, view=view_index)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        img_preds = []
        img_paths = []
        for name, out in output.items():
            out = to_cuda(out, batch.latent_index.device) # only add light probe involves interaction between out and batch
            img_pred = Visualizer.generate_image(out, batch, type)
            if cfg.vis_rotate_light:
                img_path_light = os.path.join(os.path.dirname(img_path), name + "_" + os.path.basename(img_path))
            else:
                img_path_light = os.path.join(os.path.dirname(img_path), name, os.path.basename(img_path))
            img_paths.append(img_path_light)
            img_preds.append(img_pred)

        parallel_execution(img_paths, img_preds, action=save_image)

    def summarize(self):
        for type in self.types:
            result_dir = os.path.dirname(self.img_path).format(type=type.name.lower(), view=self.view_index, frame=self.frame_index)
            for light in cfg.test_light:
                result_str = join(result_dir, light)
                if cfg.vis_rotate_light:
                    result_str = f'"{result_str}-*{cfg.vis_ext}"'
                else:
                    result_str = f'"{result_str}/*{cfg.vis_ext}"'
                Visualizer.generate_video(result_str)
