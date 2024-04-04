from enum import Enum, auto
import os
import torch
import numpy as np
from os.path import join
from copy import deepcopy
from termcolor import colored

from lib.config import cfg
from lib.config.config import Output
from lib.utils.log_utils import log, run
from lib.utils.base_utils import dotdict
from lib.utils.net_utils import normalize
from lib.utils.color_utils import colormap
from lib.utils.data_utils import save_image
from lib.utils.relight_utils import linear2srgb, add_light_probe
from lib.utils.sem_utils import get_schp_palette_tensor_float, semantics_to_color


class Visualizer:
    def __init__(self):
        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_eval_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
        f_intv = cfg.test.frame_sampler_interval
        v_intv = cfg.test.view_sampler_interval

        self.frames = [i, i + ni * i_intv, f_intv * i_intv]
        all_views = list(range(len(np.load(join(cfg.test_dataset.data_root, cfg.test_dataset.ann_file), allow_pickle=True).item()['cams']['K'])))
        all_views = cfg.test_view or all_views
        self.views = all_views[::v_intv]

        self.types = [k for k in Output if cfg[f'vis_{k.name.lower()}_map']]
        self.types = self.types or [Output.Rendering]  # with default value
        self.prepare_result_paths()

        log(f'output: {colored(self.result_dir, "blue")}')
        log(f'types: {colored([t.name.lower() for t in self.types], "blue")}')
        log(f'views: {colored(self.views, "blue")}')
        log(f'frames (start, end, intv): {colored(self.frames, "blue")}')

    def prepare_result_paths(self):
        img_path = f'{cfg.result_dir}/{{type}}/frame{{frame:04d}}_view{{view:04d}}{cfg.vis_ext}'
        img_gt_path = os.path.splitext(img_path)[0] + '_gt' + os.path.splitext(img_path)[1]
        img_loss_path = os.path.splitext(img_path)[0] + '_loss' + os.path.splitext(img_path)[1]
        self.img_path = img_path
        self.img_gt_path = img_gt_path
        self.img_loss_path = img_loss_path
        self.result_dir = os.path.dirname(img_path)

    @staticmethod
    def generate_image(output: dotdict, batch: dotdict, type: Output = Output.Rendering):
        H, W = batch.meta.H.item(), batch.meta.W.item()

        if type == Output.Normal:  # visualize normal on the surface

            norm = output.norm_map[0]
            norm = normalize(norm)
            norm = norm @ batch.cam_R[0].mT
            norm[..., 1] *= -1
            norm[..., 2] *= -1
            norm = norm * 0.5 + 0.5
            norm = norm * output.acc_map[0, ..., None]  # norm is different when blending
            rgb_map = norm

            if 'norm' in batch:
                norm = batch.norm[0]
                norm = normalize(norm)
                norm = norm @ batch.cam_R[0].mT
                norm[..., 1] *= -1
                norm[..., 2] *= -1
                norm = norm * 0.5 + 0.5
                norm = batch.msk[0, ..., None] * norm
                rgb_gt = norm

        elif type == Output.Semantic:  # visualize semantic map (NOTE: deprecated)

            palette = get_schp_palette_tensor_float(device=output.sem_map.device)
            sem = semantics_to_color(output.sem_map[0].argmax(dim=-1), palette)
            sem = output.acc_map[0, ..., None] * sem
            rgb_map = sem

            if 'sem' in batch:
                sem = semantics_to_color(batch.sem[0].argmax(dim=-1), palette)
                sem = batch.msk[0, ..., None] * sem
                rgb_gt = sem

        elif type == Output.Feature:  # visualize (optimized) feature map (NOTE: deprecated)

            feat = output.feat_map[0, ..., :3]
            # feat = (feat - feat.min()) / (feat.max() - feat.min())
            feat = output.acc_map[0, ..., None] * feat
            rgb_map = feat

        elif type == Output.Alpha:  # visualize depth map (ray occupancy accumulation) or just shadow coefficients (sphere tracing)
            rgb_map = output.acc_map[0, ..., None].expand(*output.acc_map.shape[1:], 3)  # 0 - 1
            if 'msk' in batch:
                rgb_gt = batch.msk[0, ..., None].expand(*batch.msk.shape[1:], 3)  # 0 - 1

        elif type == Output.Depth:  # visualize depth map
            if cfg.vis_median_depth:
                depth_map = output.median_map[0]
            else:
                depth_map = output.depth_map[0]
            # def depth_curve_fn(x): return torch.log(x + torch.finfo(torch.float32).eps)
            # depth_map = depth_curve_fn(depth_map)
            percentile = 0.01
            min_clip = cfg.min_clip # min depth should not be too large
            percentile_number = int(percentile * depth_map.numel())
            depth_min = depth_map[output.acc_map[0].bool()].ravel().topk(percentile_number, largest=False)[0].max()  # a simple version of percentile
            depth_max = depth_map[output.acc_map[0].bool()].ravel().topk(percentile_number, largest=True)[0].min()  # a simple version of percentile
            depth_min = depth_min.clip(None, min_clip)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            depth_map = depth_map.clip(0, 1)
            depth_map = depth_map[..., None].expand(depth_map.shape + (3,))
            rgb_map = depth_map

        elif type == Output.Shading:  # visualize shading (already normalized and sRGB)
            # shade_map = linear2srgb(output.shade_map[0])  # eH, eW, P, 3
            depth_map = output.shade_map[0]  # eH, eW, P, 3
            if cfg.normalize_shading:
                percentile = 0.005
                percentile_number = int(percentile * depth_map.numel())
                depth_min = depth_map.ravel().topk(percentile_number, largest=False)[0].max()  # a simple version of percentile
                depth_max = depth_map.ravel().topk(percentile_number, largest=True)[0].min()  # a simple version of percentile
                depth_map = depth_map / depth_max
            rgb_map = depth_map  # change naming

        elif type == Output.Albedo:
            if cfg.tonemapping_albedo:
                rgb_map = linear2srgb(output.albedo_map[0])  # maybe move this around?
            else:
                rgb_map = output.albedo_map[0]  # maybe move this around?

        elif type == Output.Roughness:
            rgb_map = output.roughness_map[0, ..., None].expand(*output.roughness_map.shape[1:], 3)
            # rgb_map = rgb_map / rgb_map.max() # make it visible

        elif type == Output.Surface:
            # use bigpose bound to make these coordinates in ndc
            rgb_map = output.cpts_map[0] if 'cpts_map' in output else output.surf_map[0]
            rgb_map = (rgb_map - batch.tbounds[0, 0:1]) / (batch.tbounds[0, 1:2] - batch.tbounds[0, 0:1])
            rgb_map = output.acc_map[0, ..., None] * rgb_map  # just...

        elif type == Output.Residual:
            # use bigpose bound to make these coordinates in ndc
            depth_map = (output.cpts_map[0] - output.bpts_map[0])
            percentile = 0.005
            percentile_number = int(percentile * depth_map.numel())
            depth_min = depth_map.ravel().topk(percentile_number, largest=False)[0].max()  # a simple version of percentile
            depth_max = depth_map.ravel().topk(percentile_number, largest=True)[0].min()  # a simple version of percentile
            depth_map = depth_map / depth_max
            # rgb_map = output.acc_map[0, ..., None] * depth_map.norm(dim=-1, keepdim=True).expand(depth_map.shape)  # just...
            rgb_map = output.acc_map[0, ..., None] * depth_map  # just...

        elif type == Output.Rendering:  # type == 'rendering'
            # visualize rgb map (default) (valid when relighting)
            rgb_map = output.rgb_map[0]
            if 'rgb' in batch:
                rgb_gt = batch.rgb[0]

        elif type == Output.Envmap:
            rgb_map = output.envmap.probe[0]

        elif type == Output.Specular:
            # visualize rgb map (default) (valid when relighting)
            depth_map = output.spec_map[0]
            if cfg.normalize_specular:
                percentile = 0.005
                percentile_number = int(percentile * depth_map.numel())
                depth_min = depth_map.ravel().topk(percentile_number, largest=False)[0].max()  # a simple version of percentile
                depth_max = depth_map.ravel().topk(percentile_number, largest=True)[0].min()  # a simple version of percentile
                depth_map = depth_map / depth_max
            rgb_map = depth_map  # change naming

        else:
            raise NotImplementedError(f'Not implemented output type: {type}')

        if rgb_map.ndim == 2:
            mask_at_box = batch.mask_at_box[0]
            mask_at_box = mask_at_box.reshape(H, W)
            mask_at_box = mask_at_box.nonzero(as_tuple=True)
            img_pred = rgb_map.new_ones(H, W, rgb_map.shape[-1]) * cfg.bg_brightness
            img_pred[mask_at_box] = rgb_map
        else:
            img_pred = rgb_map

        # add the light probe to the upper left corner when rendering with ground
        if cfg.probe_size_ratio > 0 and 'envmap' in output and type != Output.Envmap and output.envmap is not None:
            img_pred = add_light_probe(img_pred[None], output.envmap.probe, batch, cfg)[0]  # whatever

        if cfg.store_alpha_channel and type != Output.Envmap:
            mask_at_box = batch.mask_at_box[0]
            mask_at_box = mask_at_box.reshape(H, W)
            mask_at_box = mask_at_box.nonzero(as_tuple=True)
            acc_map = output.acc_map[0, ..., None]
            alpha = rgb_map.new_zeros(H, W, acc_map.shape[-1])
            alpha[mask_at_box] = acc_map  # fix alpha channel
            img_pred = torch.cat([img_pred, alpha], dim=-1)

        if 'rgb_gt' in locals() and cfg.store_ground_truth:
            if cfg.store_alpha_channel:
                if 'msk' in batch:
                    rgb_gt = torch.cat([rgb_gt, batch.msk[0, ..., None]], dim=-1)
            if rgb_gt.ndim == 2:
                img_gt = rgb_gt.new_ones(H, W, rgb_gt.shape[-1]) * cfg.bg_brightness
                img_gt[mask_at_box] = rgb_gt
            else:
                img_gt = rgb_gt

        if 'orig_H' in batch and 'orig_W' in batch:
            img_pred = Visualizer.fill_image(img_pred, batch.orig_H.item(), batch.orig_W.item(), batch.crop_bbox[0])

            if 'img_gt' in locals():
                img_gt = Visualizer.fill_image(img_gt, batch.orig_H.item(), batch.orig_W.item(), batch.crop_bbox[0])

        vis_ret = [img_pred.detach().cpu().numpy()]
        if 'img_gt' in locals():
            vis_ret.append(img_gt=img_gt.detach().cpu().numpy())
        if 'img_gt' in locals() and cfg.store_image_error:
            # l1 difference
            img_loss = (img_pred - img_gt).pow(2).sum(dim=-1).clip(0, 1)[..., None].expand(img_pred.shape)
            vis_ret.append(img_loss.detach().cpu().numpy())

        if len(vis_ret) == 1:
            return vis_ret[0]
        else:
            return vis_ret

    @staticmethod
    def fill_image(img, orig_H, orig_W, bbox):
        full_img = img.new_ones(orig_H, orig_W, 3) * cfg.bg_brightness
        height = bbox[1, 1] - bbox[0, 1]
        width = bbox[1, 0] - bbox[0, 0]
        full_img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]] = img[:height, :width]
        return full_img

    def visualize(self, output: dotdict, batch: dotdict):
        # Hacky...
        for type in self.types:
            ret = self.visualize_single_type(output, batch, type)
        return ret

    def visualize_single_type(self, output: dotdict, batch: dotdict, type: Output = Output.Rendering):
        # this executes one pass of visualization

        vis_ret = Visualizer.generate_image(output, batch, type)

        view_index = batch.meta.view_index.item()
        frame_index = batch.meta.frame_index.item()
        self.view_index = view_index  # for generating video
        self.frame_index = frame_index  # for generating video

        img_path = self.img_path.format(type=type.name.lower(), frame=frame_index, view=view_index)
        img_gt_path = self.img_gt_path.format(type=type.name.lower(), frame=frame_index, view=view_index)
        img_loss_path = self.img_loss_path.format(type=type.name.lower(), frame=frame_index, view=view_index)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        if isinstance(vis_ret, np.ndarray):
            vis_ret = [vis_ret]

        if len(vis_ret) == 3:
            img_error = vis_ret.pop(-1)
            save_image(img_loss_path, img_error)

        if len(vis_ret) == 2:
            img_gt = vis_ret.pop(-1)
            save_image(img_gt_path, img_gt)

        if len(vis_ret) == 1:
            img_pred = vis_ret.pop(-1)
            save_image(img_path, img_pred)

        return

    @staticmethod
    def generate_video(result_str):
        if not cfg.store_video_output: return
        output = result_str[1:].split('*')[0][:-1] + '.mp4'  # remove ", remove *, remove / and remove -"
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-framerate', cfg.fps,
            '-f', 'image2',
            '-pattern_type', 'glob',
            '-nostdin',
            '-y',
            '-r', cfg.fps,
            '-i', result_str,
            '-c:v', 'libx264',
            '-crf', '17',
            '-pix_fmt', 'yuv420p',
            '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',
            output,
        ]
        try:
            run(cmd)
            log(f'video generated: {colored(output, "blue")}')
        except:
            from lib.utils.log_utils import print_colorful_stacktrace
            print_colorful_stacktrace()
            pass  # continue execution without interruption

    def summarize(self):
        for type in self.types:
            result_dir = os.path.dirname(self.img_path).format(type=type.name.lower(), view=self.view_index, frame=self.frame_index)
            result_str = f'"{result_dir}/*{cfg.vis_ext}"'
            Visualizer.generate_video(result_str)
