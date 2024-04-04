import os
import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg

from lib.utils.color_utils import *
from smplx.lbs import batch_rodrigues
from collections import namedtuple


def anneal_loss_weight(weight: float, gamma: float, iter: int, mile: int):
    # exponentially anneal the loss weight
    return weight * gamma ** min(iter / mile, 1)


def gaussian_entropy_relighting4d(albedo_pred):
    albedo_entropy = 0
    for i in range(3):
        channel = albedo_pred[..., i]
        hist = GaussianHistogram(15, 0., 1., sigma=torch.var(channel))
        h = hist(channel)
        if h.sum() > 1e-6:
            h = h.div(h.sum()) + 1e-6
        else:
            h = torch.ones_like(h)
        albedo_entropy += torch.sum(-h * torch.log(h))
    return albedo_entropy


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=sigma.device).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma)**2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=1)
        return x


def gaussian_entropy(x: torch.Tensor, *args, **kwargs):
    eps = 1e-6
    hps = 1e-9
    h = gaussian_histogram(x, *args, **kwargs)
    # h = (h / (h.sum(dim=0) + hps)).clip(eps)  # 3,
    # entropy = (-h * h.log()).sum(dim=0).sum(dim=0)  # per channel entropy summed
    entropy = 0
    for i in range(3):
        hi = h[..., i]
        if hi.sum() > eps:
            hi = hi / hi.sum() + eps
        else:
            hi = torch.ones_like(hi)
        entropy += torch.sum(-hi * torch.log(hi))
    return entropy


def gaussian_histogram(x: torch.Tensor, bins: int = 15, min: float = 0.0, max: float = 1.0):
    x = x.view(-1, x.shape[-1])  # N, 3
    sigma = x.var(dim=0)  # 3,
    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5)  # BIN
    x = x[None] - centers[:, None, None]  # BIN, N, 3
    x = (-0.5 * (x / sigma).pow(2)).exp() / (sigma * np.sqrt(np.pi * 2)) * delta  # BIN, N, 3
    x = x.sum(dim=1)
    return x  # BIN, 3


def reg_diff_crit(x: torch.Tensor, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    return reg(x), weight


def reg_raw_crit(x: torch.Tensor, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    n_batch, n_pts_x2, D = x.shape
    n_pts = n_pts_x2 // 2
    length = x.norm(dim=-1, keepdim=True)  # length
    vector = x / (length + 1e-8)  # vector direction (normalized to unit sphere)
    # loss_length = mse(length[:, n_pts:, :], length[:, :n_pts, :])
    loss_vector = reg((vector[:, n_pts:, :] - vector[:, :n_pts, :]))
    # loss = loss_length + loss_vector
    loss = loss_vector
    return loss, weight


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        try:
            from torchvision.models import VGG19_Weights
            self.vgg_layers = vgg.vgg19(weights=VGG19_Weights.DEFAULT).features
        except ImportError:
            self.vgg_layers = vgg.vgg19(pretrained=True).features

        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        '''
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        '''

        self.layer_name_mapping = {'3': "relu1", '8': "relu2"}

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name == '8':
                break
        LossOutput = namedtuple("LossOutput", ["relu1", "relu2"])
        return LossOutput(**output)


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, x, target):
        x_feature = self.model(x[:, 0:3, :, :])
        target_feature = self.model(target[:, 0:3, :, :])

        feature_loss = (
            self.l1_loss(x_feature.relu1, target_feature.relu1) +
            self.l1_loss(x_feature.relu2, target_feature.relu2)) / 2.0

        l1_loss = self.l1_loss(x, target)
        l2_loss = self.mse_loss(x, target)

        loss = feature_loss + l1_loss + l2_loss

        return loss


def eikonal(x: torch.Tensor, th=1.0) -> torch.Tensor:
    return ((x.norm(dim=-1) - th)**2).mean()


def sdf_mask_crit(ret, batch):
    msk_sdf = ret['msk_sdf']
    msk_label = ret['msk_label']

    alpha = 50
    alpha_factor = 2
    alpha_milestones = [10000, 20000, 30000, 40000, 50000]
    for milestone in alpha_milestones:
        if batch['iter_step'] > milestone:
            alpha = alpha * alpha_factor

    msk_sdf = -alpha * msk_sdf
    mask_loss = F.binary_cross_entropy_with_logits(msk_sdf, msk_label) / alpha

    return mask_loss


def cross_entropy(x: torch.Tensor, y: torch.Tensor):
    # x: unormalized input logits
    # channel last cross entropy loss
    x = x.view(-1, x.shape[-1])  # N, C
    y = y.view(-1, y.shape[-1])  # N, C
    return F.cross_entropy(x, y)


def huber(x: torch.Tensor, y: torch.Tensor):
    return F.huber_loss(x, y, reduction='mean')


def smoothl1(x: torch.Tensor, y: torch.Tensor):
    return F.smooth_l1_loss(x, y)


def mse(x: torch.Tensor, y: torch.Tensor):
    return ((x.float() - y.float())**2).mean()


def dot(x: torch.Tensor, y: torch.Tensor):
    return (x * y).sum(dim=-1)


def l1(x: torch.Tensor, y: torch.Tensor):
    return l1_reg(x - y)


def l2(x: torch.Tensor, y: torch.Tensor):
    return l2_reg(x - y)


def l1_reg(x: torch.Tensor):
    return x.abs().sum(dim=-1).mean()


def l2_reg(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=-1).mean()


def mIoU_loss(x: torch.Tensor, y: torch.Tensor):
    I = (x * y).sum(-1).sum(-1)
    U = (x + y).sum(-1).sum(-1) - I
    mIoU = (I / U).mean()
    return 1 - mIoU


def reg(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1).mean()


def thresh(x: torch.Tensor, a: torch.Tensor, eps: float = 1e-8):
    return 1 / (l2(x, a) + eps)


def elastic_crit(jac):
    """
    resd_jacobian: n_batch, n_point, 3, 3
    """
    # !: CUDA IMPLEMENTATION OF SVD IS EXTREMELY SLOW
    # old_device = jac.device
    # jac = jac.cpu()
    # svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, and hence we cannot compute backward. Please use torch.svd(compute_uv=True)
    _, S, _ = torch.svd(jac, compute_uv=True)
    # S = S.to(old_device)
    log_svals = torch.log(torch.clamp(S, min=1e-6))
    elastic_loss = (log_svals**2).mean()
    return elastic_loss
