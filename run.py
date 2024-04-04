import time
import tqdm
import torch
import numpy as np

from lib.utils.net_utils import load_network
from lib.utils.data_utils import to_cuda
from lib.evaluators import make_evaluator
from lib.visualizers import make_visualizer
from lib.networks.renderer import make_renderer
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.config import cfg, args
from lib.utils.log_utils import log

import cv2
cv2.setNumThreads(1)
cfg.train.num_workers = 0  # no multi-process dataloading needed when visualizing


@torch.no_grad()
def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


@torch.no_grad()
def run_network():
    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()
    renderer = make_renderer(cfg, network)

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)

        torch.cuda.synchronize()
        start = time.time()
        output = renderer.render(batch)
        torch.cuda.synchronize()
        total_time += time.time() - start

    log(total_time / len(data_loader))


@torch.no_grad()
def run_evaluate():
    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        output = renderer.render(batch)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


@torch.no_grad()
def run_visualize():
    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()
    diffs = []

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch, 'cuda')
        output = renderer.render(batch)

        if 'diff' in output:
            diffs.append(output.diff)
            del output.diff
        visualizer.visualize(output, batch)
    visualizer.summarize()

    if len(diffs):
        log(f'###################{cfg.exp_name}###################', 'green')
        log(f'Net work rendering time: {np.mean(diffs)}', 'green')


if __name__ == '__main__':
    try:
        globals()['run_' + args.type]()
    except:
        import pdbr
        pdbr.post_mortem()
