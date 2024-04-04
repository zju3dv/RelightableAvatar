import os
import torch

from collections import deque, defaultdict
from torch.utils.tensorboard import SummaryWriter

from lib.config.config import cfg
from lib.utils.base_utils import dotdict, default_dotdict
from lib.utils.log_utils import log


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.value = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value
        self.value = value

    @property
    def latest(self):
        d = torch.tensor(list(self.deque))
        return d[-1].item()

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.float().median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.float().mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def val(self):
        return self.value


class Recorder(object):
    def __init__(self, cfg):
        if cfg.local_rank > 0:
            return

        log_dir = cfg.record_dir
        if not cfg.resume:
            log(f'removing training record: {log_dir}', 'red')
            os.system(f'rm -rf {log_dir}')
        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.record_stats = default_dotdict(SmoothedValue)

        # images
        self.image_stats = default_dotdict(object)
        if 'process_' + cfg.task in globals():
            self.processor = globals()['process_' + cfg.task]
        else:
            self.processor = None

    def update_record_stats(self, record_stats: dotdict):
        if cfg.local_rank > 0:
            return
        for k, v in record_stats.items():
            self.record_stats[k].update(v)

    def update_image_stats(self, image_stats: dotdict):
        if cfg.local_rank > 0:
            return
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v

    def record(self, prefix, step=-1, record_stats: dotdict = None, image_stats: dotdict = None):
        if cfg.local_rank > 0:
            return

        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        record_stats = record_stats if record_stats else self.record_stats

        for k, v in record_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            return
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        if cfg.local_rank > 0:
            return
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        if cfg.local_rank > 0:
            return
        self.step = scalar_dict['step']

    @property
    def log_stats(self):
        if cfg.local_rank > 0:
            return
        log_stats = dotdict()
        log_stats.epoch = str(self.epoch)
        log_stats.step = str(self.step)
        for k, v in self.record_stats.items():
            if isinstance(v, SmoothedValue):
                log_stats[k] = f'{v.avg:.6f}'
            else:
                log_stats[k] = v
        log_stats.lr = f'{self.record_stats.lr.val:.6f}'
        log_stats.data = f'{self.record_stats.data.val:.4f}'
        log_stats.batch = f'{self.record_stats.batch.val:.4f}'
        log_stats.max_mem = f'{self.record_stats.max_mem.val:.0f}'
        return log_stats

    def __str__(self):
        return '  '.join([k + ': ' + v for k, v in self.log_stats.items()])


def make_recorder(cfg):
    recorder = Recorder(cfg)

    return recorder
