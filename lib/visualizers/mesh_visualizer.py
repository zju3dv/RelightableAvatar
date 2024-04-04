import os

from lib.config import cfg
from lib.utils.data_utils import export_dotdict, export_mesh
from lib.utils.log_utils import log, print_colorful_stacktrace, run
from . import base_visualizer


class Visualizer(base_visualizer.Visualizer):
    def __init__(self):
        self.result_dir = 'data/animation/{}/{}'.format(cfg.task, cfg.exp_name)
        log('the results are saved at {}'.format(self.result_dir), 'yellow')

    def visualize(self, output, batch):
        result_dir = self.result_dir

        if (batch.meta.latent_index.item() == -1 and cfg.vis_tpose_mesh and cfg.track_tpose_mesh) or cfg.vis_can_mesh:
            os.makedirs(result_dir, exist_ok=True)
            meta_path = os.path.join(result_dir, 'can_mesh.npz')
            mesh_path = os.path.join(result_dir, 'can_mesh.ply')
        elif cfg.vis_posed_mesh or cfg.vis_tpose_mesh:
            if cfg.track_tpose_mesh:
                result_dir = os.path.join(result_dir, 'track_mesh')
            elif cfg.vis_posed_mesh:
                result_dir = os.path.join(result_dir, 'posed_mesh')
            elif cfg.vis_tpose_mesh:
                result_dir = os.path.join(result_dir, 'tpose_mesh')

            os.makedirs(result_dir, exist_ok=True)
            frame_index = batch.meta.frame_index.item()
            view_index = batch.meta.view_index.item()
            meta_path = os.path.join(result_dir, '{:04d}.npz'.format(frame_index))
            mesh_path = os.path.join(result_dir, '{:04d}.ply'.format(frame_index))

        export_dotdict(output, meta_path)
        export_mesh(output.verts, output.faces, filename=mesh_path)

        try:
            run(f'blender --background --python-expr \"import sys; sys.path.append(\'.\'); from lib.utils.blender_utils import replace_weights; replace_weights(\'{meta_path}\', \'{meta_path}\')\"')
        except Exception as e:
            log('blender not found or returned error, will use SMPL blend weights and they might be janky. Maybe try to install blender from https://www.blender.org/download/. Use the log above for more info.', 'red')

    def prepare_result_paths(self): pass

    def update_result_dir(self): pass

    def summarize(self): pass
