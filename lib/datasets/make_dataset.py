import cv2
import time
import torch
import importlib
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate, default_convert
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler

from lib.config.config import cfg
from lib.datasets import samplers

torch.multiprocessing.set_sharing_strategy('file_system')
cv2.setNumThreads(1)


def _dataset_factory(cfg, is_train):
    if is_train:
        module = cfg.train_dataset_module
        args = cfg.train_dataset
    else:
        module = cfg.test_dataset_module
        args = cfg.test_dataset
    # __import__ is like import ...
    # __import_module__ is like from ... import ...
    # dataset = __import__(module, fromlist=[None]).Dataset(**args)
    # The __import__ function will return the top level module of a package, unless you pass a nonempty fromlist argument:
    # MARK: imp.load_source breaks the typing in Dataset, occ_dataset.Dataset would not == occ_dataset.Dataset.Dataset
    # switching to import lib can solve this problem
    # dataset = __import__(module, fromlist=[None]).Dataset(**args)
    dataset = importlib.import_module(module).Dataset(**args)
    return dataset


def make_dataset(cfg, is_train=True):
    dataset = _dataset_factory(cfg, is_train)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed, is_train):
    if not is_train and cfg.test.sampler == 'FrameSampler':
        sampler = samplers.FrameSampler(dataset)
        return sampler
    if not is_train and cfg.test.sampler == 'MeshFrameSampler':
        sampler = samplers.MeshFrameSampler(dataset)
        return sampler
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size, drop_last, sampler_meta)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    cv2.setNumThreads(1)  # MARK: OpenCV undistort is why all cores are taken
    # previous randomness issue might just come from here
    if cfg.fix_random:
        np.random.seed(worker_id)
    else:
        np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1) -> DataLoader:
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset = make_dataset(cfg, is_train)
    sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train)
    num_workers = min(cfg.train.num_workers, max_iter) if max_iter > 0 else cfg.train.num_workers
    prefetch_factor = cfg.prefetch_factor if num_workers > 1 else (None if torch.__version__ >= '2' else 2)
    pin_memory = cfg.pin_memory
    collate = default_collate if cfg.collate else default_convert
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             worker_init_fn=worker_init_fn,
                             collate_fn=collate,
                             pin_memory=pin_memory,
                             prefetch_factor=prefetch_factor
                             )

    return data_loader
