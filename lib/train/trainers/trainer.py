import tqdm
import time
import torch
import datetime
import numpy as np
from typing import List
from torch.optim import Optimizer

from lib.config import cfg
from lib.train.recorder import Recorder
from lib.utils.base_utils import dotdict
from lib.utils.prof_utils import profiler_step
from lib.utils.data_utils import add_iter_step, to_cuda
from lib.utils.log_utils import log, update_log_stats, print_colorful_stacktrace

from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = DDP(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=cfg.find_unused_parameters,
            )
        self.network = network
        self.device = device
        self.local_rank = cfg.local_rank

    def reduce_record_stats(self, record_stats: dotdict):
        reduced_stats = dotdict()
        for k, v in record_stats.items():
            if isinstance(v, torch.Tensor):
                reduced_stats[k] = v.item()  # MARK: will cause sync
            else:
                reduced_stats[k] = v
        return reduced_stats

    # NOTE: this is now a generator instead of simple function, to preserve the local variables
    def train(self, epoch, data_loader, optimizers: List[Optimizer], recorder: Recorder):
        ep_iter = cfg.ep_iter
        self.network.train()
        end = time.perf_counter()

        for index, batch in enumerate(data_loader):
            if hasattr(recorder, 'step'):
                iteration = recorder.step + 1
            else:
                iteration = 1 # the start of the training process

            batch = add_iter_step(batch, iteration)  # DotDict
            batch = to_cuda(batch)  # will also return as DotDict

            data_time = time.perf_counter() - end  # data time
            output, loss, record_stats, image_stats = self.network(batch)

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss = loss.mean()
            loss.backward()  # where the actual work is done

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.clip_grad_norm)
            torch.nn.utils.clip_grad_value_(self.network.parameters(), cfg.clip_grad_value)

            for optimizer in optimizers:
                optimizer.step()  # all optimizers on all GPUs should perform backward step?

            if cfg.local_rank > 0:
                if iteration % ep_iter == 0:
                    yield
                continue

            recorder.step += 1
            if iteration % cfg.log_interval == 0 or iteration % ep_iter == 0:
                record_stats = self.reduce_record_stats(record_stats)

                # data recording stage: loss_stats, time, image_stats
                batch_time = time.perf_counter() - end
                lr = optimizer.param_groups[0]['lr']
                max_mem = torch.cuda.max_memory_allocated() / 2**20

                record_stats.data = data_time
                record_stats.batch = batch_time
                record_stats.lr = lr
                record_stats.max_mem = max_mem
                recorder.update_record_stats(record_stats)

                eta_seconds = recorder.record_stats.batch.global_avg * (cfg.train.epoch * ep_iter - recorder.step)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_stats = dotdict()
                log_stats.name = cfg.exp_name
                log_stats.eta = eta_string
                log_stats.update(recorder.log_stats)

                update_log_stats(log_stats, table_row_limit=cfg.table_row_limit)

            if iteration % cfg.record_interval == 0 or iteration % ep_iter == 0:
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

            profiler_step()
            end = time.perf_counter()

            if iteration % ep_iter == 0:
                yield

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            batch = add_iter_step(batch, epoch * cfg.ep_iter)  # DotDict
            batch = to_cuda(batch)
            with torch.no_grad():

                output, loss, loss_stats, image_stats = self.network(batch)

                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_record_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        log(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)

