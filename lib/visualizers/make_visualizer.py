import os
import importlib
from .base_visualizer import Visualizer

def make_visualizer(cfg) -> Visualizer:
    module = cfg.visualizer_module
    visualizer = importlib.import_module(module).Visualizer()
    return visualizer
