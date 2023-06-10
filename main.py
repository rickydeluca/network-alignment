import argparse
import math
import random
from itertools import islice

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn import Module


def parse_args():
    """
    Parse the terminal arguments.
    """
    parser = argparse.ArgumentParser(description='Choose the algorithm for the netwok alignment and the data to build the networks.')
    parser.add_argument('-a','--alg', type=str, default=None,
                    help='Algorithm for the network alignment. Choose from: ["final"].')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='What dataset to use to generate the networks. Choose from: [""].')
    parser.add_argument('-r', '--reproducibility', type=bool, default=True,
                help='Activate reproducibility. (default: True).')
    return parser.parse_args()


def read_input(args):
    """
    Given the parsed argument list, verify their correcteness and return each of them.
    """
    valid_algs = ["isorank"]
    valid_datasets = ["partial_synthetic"]

    if args.alg is None or args.alg not in valid_algs:
        raise ValueError(f"{args.alg} is not a valid algorithm!")
    
    if args.dataset is None or args.dataset not in valid_datasets:
        raise ValueError(f"{args.datasate} is not a valid algorithm!")
    
    return args.alg, args.dataset, args.reproducibility


def set_reproducibility(reproducibility):
    """
    Set seeds and options for reproducibility.
    """
    if reproducibility:
        np.random.seed(42)
        random.seed(42)
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # Read input.
    args = parse_args()
    alg, dataset, reproducibility = read_input(args)

    # Reproducibility.
    set_reproducibility(reproducibility)

