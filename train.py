# fmt: off
# cfg should be imported first before importing torch, to avoid cfg mismatch
# and also to avoid loading cuda_driver before proper CUDA_VISIBLE_DEVICES is set
from lib.config import cfg, args
# fmt: on

from lib.networks import make_network
from lib.evaluators import make_evaluator
from lib.datasets import make_data_loader
from lib.utils.log_utils import log, print_colorful_stacktrace
from lib.utils.net_utils import load_model, save_model, load_network, fix_random, number_of_params
from lib.utils.prof_utils import setup_profiling, profiler_start, profiler_step, profiler_stop
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler

import os
import torch
import torch.multiprocessing
import torch.distributed as dist

from torch import nn
from rich.pretty import pprint
from easyvolcap.utils.console_utils import *

@catch_throw
def train(cfg, network: nn.Module):
    setup_profiling(cfg.profiling)
    fix_random(cfg.fix_random)
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    optims = [optimizer]
    begin_epoch = load_model(network,
                             optims,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             epoch=cfg.train.load_epoch,
                             resume=cfg.resume,
                             load_others=cfg.load_others,
                             )
    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter * (cfg.train.epoch - begin_epoch))
    val_loader = make_data_loader(cfg, is_train=False)

    fix_random(cfg.fix_random)
    profiler_start()

    do_train = trainer.train(begin_epoch, train_loader, optims, recorder)
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        network.train()  # selet training mode
        try:
            next(do_train)  # actual training
        except RuntimeError as e:
            print_colorful_stacktrace()
            import pdbr; pdbr.post_mortem()  # break on the last exception's stack for inpection
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            # for ZeroRedundancyOptimizer, we need to consolidate the gradients before saving the state dicts
            save_model(network, optims, scheduler, recorder, cfg.trained_model_dir, epoch, rank=cfg.local_rank, distributed=cfg.distributed)

        if (epoch + 1) % cfg.save_latest_ep == 0:
            save_model(network, optims, scheduler, recorder, cfg.trained_model_dir, epoch, latest=True, rank=cfg.local_rank, distributed=cfg.distributed)

        if (epoch + 1) % cfg.eval_ep == 0:
            try:
                trainer.val(epoch, val_loader, evaluator, recorder)  # sometimes during early testing stage, evaluation is not implemented
            except Exception as e:
                print_colorful_stacktrace()
                log(f"exception when evaluating: {type(e)}: {e}, check your eval impl", 'red')
                # do not disrupt training even if validation raised an exception

    profiler_stop()
    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    fix_random(cfg.fix_random)
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        log(f'local rank: {cfg.local_rank}, world_size: {cfg.world_size}')
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    network = make_network(cfg)

    if cfg.print_network:
        nop = number_of_params(network)
        log('')
        pprint(network)
        log(f'number of parameters: {nop}({nop / 1e6:.2f}M)')

    if cfg.dry_run:
        return

    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    if cfg.detect_anomaly:
        with torch.autograd.detect_anomaly():
            main()
    else:
        main()
