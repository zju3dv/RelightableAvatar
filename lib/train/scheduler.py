from collections import Counter
from .optimizers.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR, WarmupExponentialLR


def make_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs,
                                  gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'warmup_exponential':
        scheduler = WarmupExponentialLR(optimizer,
                                        warmup_factor=cfg_scheduler.warmup_factor,
                                        warmup_epochs=cfg_scheduler.warmup_epochs,
                                        warmup_method=cfg_scheduler.warmup_method,
                                        decay_epochs=cfg_scheduler.decay_epochs,
                                        gamma=cfg_scheduler.gamma)
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
    elif cfg_scheduler.type == 'warmup_exponential':
        scheduler.warmup_factor = cfg_scheduler.warmup_factor
        scheduler.warmup_epochs = cfg_scheduler.warmup_epochs
        scheduler.warmup_method = cfg_scheduler.warmup_method
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
    scheduler.gamma = cfg_scheduler.gamma
