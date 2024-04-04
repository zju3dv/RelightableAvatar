import torch
from torch import nn
from termcolor import colored
from lib.utils.log_utils import log
from .optimizers.radam import RAdam
from torch.distributed.optim import ZeroRedundancyOptimizer


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net: nn.Module, lr=None, eps=None, weight_decay=None):
    params = []
    eps = cfg.train.eps if eps is None else eps
    lr = cfg.train.lr if lr is None else lr
    weight_decay = cfg.train.weight_decay if weight_decay is None else weight_decay

    special = []
    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue

        v_lr = lr
        v_eps = eps
        v_weight_decay = weight_decay

        keys = key.split('.')
        for item in keys:
            if item in cfg.train.lr_table:
                v_lr = cfg.train.lr_table[item]
                special.append(f'{key}: {colored(f"{v_lr:g}", "magenta")}')
                break
        for item in keys:
            if item in cfg.train.eps_table:
                v_eps = cfg.train.eps_table[item]
                break
        for item in keys:
            if item in cfg.train.weight_decap_table:
                v_weight_decay = cfg.train.weight_decap_table[item]
                break
        params += [{"params": [value], "lr": v_lr, "weight_decay": v_weight_decay, 'eps': v_eps}]

    log(f'default learning rate: {colored(f"{lr:g}", "magenta")}')
    if len(special):
        log(f'special learning rate loaded from lr table: \n' + '\n'.join(special))

    if 'adam' in cfg.train.optim:
        # if cfg.distributed:
        #     optimizer = ZeroRedundancyOptimizer(params, optimizer_class=_optimizer_factory[cfg.train.optim], lr=lr, weight_decay=weight_decay)
        # else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        # if cfg.distributed:
        #     optimizer = ZeroRedundancyOptimizer(params, optimizer_class=_optimizer_factory[cfg.train.optim], lr=lr)
        # else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
