"""
Dataset class and DataLoader utilities adapted from 
ThinkMatch: https://github.com/Thinklab-SJTU/ThinkMatch
"""


import random

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset

cfg = edict()   # global variable that substitue a configuration file


class GMSemiSyntheticDataset(Dataset):
    def __init__(self,
                 source_dataset,
                 noise: float=None,
                 length: int=None,
                 mode: str='train'):
        
        super(GMSemiSyntheticDataset, self).__init__()
        self.source_dataset = source_dataset    # using the base `Dataset` class
        self.noise = noise                      # > 0: add dummy edges, < 0: remove edges
        self.length = length
        self.mode = mode

    def __getitem__(self, idx):
        return self.get_pair(idx, self.noise)

    @staticmethod
    def get_pair(self, idx, noise):
        """
        Return a dictionary the all the informations
        of a source-target pair.
        - 'source' is always the graph generated from the 'source_dataset'.
        - 'target' is a random clone with noise of the source graph.
        """


def collate_fn(data):
    """
    Custom collate functino for the DataLoader.
    """
    # TODO
    # ...
    return


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False, batch_size=None, num_workers=None, seed=None):
    # set dataloader configuration
    cfg.RANDOM_SEED = seed
    cfg.BATCH_SIZE = batch_size
    cfg.DATALOADER_NUM = num_workers

    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )