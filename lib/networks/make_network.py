import importlib
from torch import nn

def make_network(cfg) -> nn.Module:
    module = cfg.network_module
    network = importlib.import_module(module).Network()
    return network
