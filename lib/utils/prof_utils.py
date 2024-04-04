import os
import torch
from lib.utils.log_utils import log
from lib.utils.base_utils import dotdict, context
from torch.profiler import profile, record_function, ProfilerActivity, schedule

context.prof_cfg = dotdict()
context.prof_cfg.enabled = False


def profiler_step():
    if context.prof_cfg.enabled:
        context.profiler.step()


def profiler_start():
    if context.prof_cfg.enabled:
        context.profiler.start()


def profiler_stop():
    if context.prof_cfg.enabled:
        context.profiler.stop()


def setup_profiling(prof_cfg):
    if prof_cfg.enabled:
        log(f"profiling results will be saved to: {prof_cfg.record_dir}", 'yellow')
        if prof_cfg.clear_previous:
            log(f'removing profiling result in: {prof_cfg.record_dir}', 'red')
            os.system(f'rm -rf {prof_cfg.record_dir}')
        profiler = profile(schedule=schedule(skip_first=prof_cfg.skip_first,
                                             wait=prof_cfg.wait,
                                             warmup=prof_cfg.warmup,
                                             active=prof_cfg.active,
                                             repeat=prof_cfg.repeat,
                                             ),
                           activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_cfg.record_dir),
                           record_shapes=prof_cfg.record_shapes,
                           profile_memory=prof_cfg.profile_memory,
                           with_stack=prof_cfg.with_stack,  # sometimes with_stack causes segmentation fault
                           with_flops=prof_cfg.with_flops,
                           with_modules=prof_cfg.with_modules
                           )
        context.profiler = profiler
        context.prof_cfg = prof_cfg
