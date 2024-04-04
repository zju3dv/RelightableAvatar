import os
from lib.config import cfg
from lib.config.config import Output
from lib.utils.data_utils import save_image
from lib.utils.base_utils import dotdict
from . import base_visualizer


class Visualizer(base_visualizer.Visualizer):
    def prepare_result_paths(self):
        img_path = f'data/pose_sequence/{cfg.exp_name}/view_{{view:04d}}/{{type}}/frame_{{frame:04d}}{cfg.vis_ext}'
        self.img_path = img_path
        self.result_dir = os.path.dirname(self.img_path)

    def visualize_single_type(self, output: dotdict, batch: dotdict, type: Output = Output.Rendering):
        img_pred = Visualizer.generate_image(output, batch, type)
        frame_index = batch.meta.frame_index.item()
        view_index = batch.meta.view_index.item()
        self.view_index = view_index  # for generating video
        self.frame_index = frame_index  # for generating video

        img_path = self.img_path.format(type=type.name.lower(), frame=frame_index, view=view_index)
            
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        save_image(img_path, img_pred)
