import os
import importlib
from .base_renderer import Renderer

def make_renderer(cfg, network) -> Renderer:
    module = cfg.renderer_module
    renderer = importlib.import_module(module).Renderer(network)
    return renderer
