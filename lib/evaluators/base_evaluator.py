import os
import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

from lib.config import cfg
from lib.utils.log_utils import log
from ..visualizers import base_visualizer


class Evaluator(base_visualizer.Visualizer):
    def __init__(self):
        if cfg.local_rank > 0:
            return

        super(Evaluator, self).__init__()

        import lpips
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.compute_lpips = lpips.LPIPS(verbose=False)

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch):
        if not cfg.eval_whole_img:
            mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
            H, W = batch['H'].item(), batch['W'].item()
            mask_at_box = mask_at_box.reshape(H, W)
            # crop the object region
            x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
            img_pred = img_pred[y:y + h, x:x + w]
            img_gt = img_gt[y:y + h, x:x + w]

        if 'crop_bbox' in batch:
            img_pred = Evaluator.fill_image(img_pred, batch)
            img_gt = Evaluator.fill_image(img_gt, batch)

        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, channel_axis=-1, data_range=1)

        return ssim

    def lpips_metric(self, img_pred, img_gt, batch):
        if not cfg.eval_whole_img:
            mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
            H, W = batch['H'].item(), batch['W'].item()
            mask_at_box = mask_at_box.reshape(H, W)
            # crop the object region
            x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
            img_pred = img_pred[y:y + h, x:x + w]
            img_gt = img_gt[y:y + h, x:x + w]

        if 'crop_bbox' in batch:
            img_pred = Evaluator.fill_image(img_pred, batch)
            img_gt = Evaluator.fill_image(img_gt, batch)

        # compute the lpips
        with torch.no_grad():
            lpips = self.compute_lpips(torch.Tensor(img_pred.transpose((2, 0, 1))[None]), torch.Tensor(img_gt.transpose((2, 0, 1)))[None])[0]
        lpips = lpips.item()

        return lpips

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        white_bkgd = cfg.bg_brightness

        if rgb_pred.ndim == 2:
            img_pred = np.zeros((H, W, 3)) + white_bkgd
            img_pred[mask_at_box] = rgb_pred
            img_gt = np.zeros((H, W, 3)) + white_bkgd
            img_gt[mask_at_box] = rgb_gt
        else:
            img_pred = rgb_pred
            img_gt = rgb_gt

        if cfg.eval_whole_img:
            rgb_pred = img_pred
            rgb_gt = img_gt

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        lpips = self.lpips_metric(rgb_pred, rgb_gt, batch)
        self.lpips.append(lpips)

        self.visualize(output, batch)

    def summarize(self):
        super(Evaluator, self).summarize()  # will save images

        result_dir = cfg.result_dir
        log('the results are saved at {}'.format(result_dir), 'yellow')

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips}
        np.save(result_path, metrics)
        mse, psnr, ssim, lpips = np.mean(self.mse), np.mean(self.psnr), np.mean(self.ssim), np.mean(self.lpips)
        mean_metrics = {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
        log('mse: {}'.format(np.mean(self.mse)))
        log('psnr: {}'.format(np.mean(self.psnr)))
        log('ssim: {}'.format(np.mean(self.ssim)))

        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []

        return mean_metrics
