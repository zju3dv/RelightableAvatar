import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.loss_utils import l1, l2, elastic_crit, eikonal, mse, l1_reg, dot, mIoU_loss, huber, cross_entropy, reg_raw_crit, anneal_loss_weight
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

        # mipnerf360 regularzation
        if 'distortion' in ret:
            dist_loss = ret['distortion'].mean()
            scalar_stats.update({'dist_loss': dist_loss})
            loss += cfg.dist_loss_weight * dist_loss

        # human-nerf optimizing monocular pose
        if 'presd' in ret:
            presd_loss = ret['presd'].view(-1, 3).norm(dim=-1).mean()
            scalar_stats.update({'presd_loss': presd_loss})
            loss += cfg.presd_loss_weight * presd_loss

        # deforming aninerf loss
        if 'oresd' in ret and ret.oresd.numel():
            oresd, weight = reg_raw_crit(ret.oresd, batch.iter_step)  # svd of jacobian elastic loss
            scalar_stats.update({'oresd': oresd})
            loss += oresd * weight

        if 'jac' in ret and ret.jac.numel():
            jac, weight = reg_raw_crit(ret.jac, batch.iter_step)  # length of difference in value of neighbor points
            scalar_stats.update({'jac': jac})
            loss += jac * weight  # TODO: remove weights

        if 'ograd' in ret and ret.ograd.numel():
            ograd, weight = reg_raw_crit(ret.ograd, batch.iter_step)  # length of difference in value of neighbor points
            scalar_stats.update({'ograd': ograd})
            loss += ograd * weight

        if 'cgrad' in ret and ret.cgrad.numel():
            cgrad, weight = reg_raw_crit(ret.cgrad, batch.iter_step)  # length of difference in value of neighbor points
            scalar_stats.update({'cgrad': cgrad})
            loss += cgrad * weight

        # anisdf loss
        if 'residuals' in ret:  # residual offset
            resd_loss = ret['residuals'].norm(dim=-1).mean()
            resd_loss_weight = anneal_loss_weight(cfg.resd_loss_weight, cfg.resd_loss_weight_gamma, batch.meta.iter_step, cfg.resd_loss_weight_milestone)
            scalar_stats.update({'resd_loss': resd_loss})
            if cfg.resd_loss_weight_gamma != 1.0:
                scalar_stats.update({'resd_loss_weight': resd_loss_weight})
            loss += resd_loss_weight * resd_loss

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

        if 'norm_map' in ret and 'norm' in batch:
            # image derivative only available when we're doing patch sampling, how to achieve this?
            norm_map = normalize(ret['norm_map'])  # world space normal B, N, 3 @ B, 3, 3
            norm = normalize(batch['norm'])  # in world space right?
            view_map = batch['ray_d']  # B, N, 3
            view_dot = dot(norm_map, -view_map).clip(0, 1)  # B, N, this serves as weight to the normal loss
            norm_loss = ((norm_map - norm).abs().sum(dim=-1) + (1 - dot(norm_map, norm))) * view_dot
            norm_loss = norm_loss.mean()
            # norm_loss = l1(norm_map, norm)
            scalar_stats.update({'norm_loss': norm_loss})
            loss += cfg.norm_loss_weight * norm_loss

        if 'sem_map' in ret and 'sem' in batch:
            sem_loss = cross_entropy(ret['sem_map'], batch['sem'])
            scalar_stats.update({'sem_loss': sem_loss})
            loss += cfg.sem_loss_weight * sem_loss

        if 'acc_map' in ret and 'msk' in batch:
            msk_loss = mIoU_loss(ret['acc_map'], batch['msk'])
            scalar_stats.update({'msk_loss': msk_loss})
            loss += cfg.msk_loss_weight * msk_loss

        if 'rgb_map' in ret:
            img_loss = mse(ret['rgb_map'], batch['rgb'])
            psnr = (1 / img_loss).log() * 10 / np.log(10)
            scalar_stats.update({'img_loss': img_loss})
            scalar_stats.update({'psnr': psnr})
            loss += cfg.img_loss_weight * img_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
