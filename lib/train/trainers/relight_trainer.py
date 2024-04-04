import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.loss_utils import l1, l2, mse, gaussian_entropy, gaussian_entropy_relighting4d, eikonal, mIoU_loss
from lib.utils.net_utils import normalize
from lib.networks.renderer import base_renderer, make_renderer


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.renderer: base_renderer.Renderer = make_renderer(cfg, net)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = dotdict()
        loss = 0

        # directional distance trainer
        if 'dd_gt' in ret and 'dd_pred' in ret:
            pred = ret.dd_pred
            gt = ret.dd_gt
            dd_loss = mse(pred, gt)
            scalar_stats.update({'dd_loss': dd_loss})
            loss += cfg.dd_loss_weight * dd_loss

        if 'di_gt' in ret and 'di_pred' in ret:
            pred = ret.di_pred
            gt = ret.di_gt
            di_loss = mse(pred, gt)
            scalar_stats.update({'di_loss': di_loss})
            loss += cfg.di_loss_weight * di_loss

        if 'pi_gt' in ret and 'pi_pred' in ret:
            pred = ret.pi_pred
            gt = ret.pi_gt
            pi_loss = mIoU_loss(pred, gt)
            scalar_stats.update({'pi_loss': pi_loss})
            loss += cfg.pi_loss_weight * pi_loss

        # anisdf loss
        if 'residuals' in ret:  # residual offset
            resd_loss = ret['residuals'].norm(dim=-1).mean()
            scalar_stats.update({'resd_loss': resd_loss})
            loss += cfg.resd_loss_weight * resd_loss

        if 'gradients' in ret:  # gradients
            gradients = ret['gradients']
            grad_loss = eikonal(gradients)
            scalar_stats.update({'grad_loss': grad_loss})
            loss += cfg.eikonal_loss_weight * grad_loss

        if 'observed_gradients' in ret and ret['observed_gradients'].numel():
            ogradients = ret['observed_gradients']
            ograd_loss = eikonal(ogradients)
            scalar_stats.update({'ograd_loss': ograd_loss})
            loss += cfg.observed_eikonal_loss_weight * ograd_loss

        if 'acc_map' in ret and 'msk' in batch:
            msk_loss = mIoU_loss(ret['acc_map'], batch['msk'])
            scalar_stats.update({'msk_loss': msk_loss})
            loss += cfg.msk_loss_weight * msk_loss

        # relighting loss
        if 'albedo' in ret:
            albedo_entropy = gaussian_entropy(ret.albedo)
            scalar_stats.update({'albedo_entropy': albedo_entropy})
            # print(ret.albedo.max())
            # if albedo_entropy == 0.0:
            #     breakpoint()
            loss += cfg.albedo_sparsity * albedo_entropy

        if 'volume_albedo' in ret:
            albedo_entropy = gaussian_entropy(ret.volume_albedo)
            scalar_stats.update({'volume_entropy': albedo_entropy})
            loss += cfg.albedo_sparsity * albedo_entropy

        if 'albedo' in ret and 'albedo_jitter' in ret:
            albedo_smooth = l1(ret['albedo'], ret['albedo_jitter'])
            scalar_stats.update({'albedo_smooth': albedo_smooth})
            loss += cfg.albedo_smooth_weight * albedo_smooth

        if 'roughness' in ret and 'roughness_jitter' in ret:
            roughness_smooth = l1(ret['roughness'], ret['roughness_jitter'])
            scalar_stats.update({'roughness_smooth': roughness_smooth})
            loss += cfg.roughness_smooth_weight * roughness_smooth

        if 'normal' in ret and 'normal_jitter' in ret:
            normal_smooth = l1(ret['normal'], ret['normal_jitter'])
            scalar_stats.update({'normal_smooth': normal_smooth})
            loss += cfg.normal_smooth_weight * normal_smooth

        if 'visibility' in ret and 'visibility_jitter' in ret:
            visibility_smooth = l1(ret['visibility'], ret['visibility_jitter'])
            scalar_stats.update({'visibility_smooth': visibility_smooth})
            loss += cfg.visibility_smooth_weight * visibility_smooth

        if 'normal' in ret and 'normal_geometry' in ret:
            normal_smooth = l2(ret['normal'], ret['normal_geometry'])
            scalar_stats.update({'normal_geometry': normal_smooth})
            loss += cfg.normal_geometry_weight * normal_smooth

        if 'visibility' in ret and 'visibility_geometry' in ret:
            visibility_smooth = l2(ret['visibility'], ret['visibility_geometry'])
            scalar_stats.update({'visibility_geometry': visibility_smooth})
            loss += cfg.visibility_geometry_weight * visibility_smooth

        if 'rgb_map' in ret:
            img_loss = mse(ret['rgb_map'], batch['rgb'])
            psnr = (1 / img_loss).log() * 10 / np.log(10)
            scalar_stats.update({'img_loss': img_loss})
            scalar_stats.update({'psnr': psnr})
            loss += cfg.img_loss_weight * img_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
