from .trainer import Trainer
import importlib


def _wrapper_factory(cfg, network):
    module = cfg.trainer_module
    network_wrapper = importlib.import_module(module).NetworkWrapper(network)
    return network_wrapper


def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return Trainer(network)
