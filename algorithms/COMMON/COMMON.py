import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from algorithms.network_alignment_model import NetworkAlignmentModel


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def dataset_to_pyg(dataset, pos_info=False, normalize=False):
    """
    Given a `Dataset` object, return the corresponding pyg graph.
    """
    # Convert NetworX to Data representation
    G = from_networkx(dataset.G)

    # Extract useful informations
    edge_index = G.edge_index
    edge_weight = G.weight if 'weight' in G.keys() else None

    # Read node/edge features    
    x = torch.tensor(dataset.features, dtype=torch.float32) if dataset.features is not None else None
    edge_attr = torch.tensor(dataset.edge_features) if dataset.edge_features is not None else None

    # Use positional informations if required
    if pos_info:
        adj_matrix = torch.tensor(nx.adjacency_matrix(dataset.G).todense(), dtype=torch.float32)
        if x is not None:
            x = torch.cat((x, adj_matrix), dim=1)

    # Normalize attributes if required
    if normalize:
        x = normalize_over_channels(x)
        edge_attr = normalize_over_channels(edge_attr)

    # Build the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

    return data


class COMMON(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        super().__init__(source_dataset, target_dataset)

        # Base settings
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.train_dict = args.train_dict
        self.groundtruth = args.groundtruth
        self.seed = args.seed
        self.device = torch.device(args.device)
        self.S = None

        # Traning settings
        self.train_epochs = args.train_epochs
        self.loss_func = args.loss_func
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.separate_backbone_lr = args.separate_backbone_lr
        self.backbone_lr = args.backbone_lr
        self.train_momentum = args.momentum
        self.lr_decay = args.lr_decay
        self.lr_step = args.lr_step
        self.epoch_iters = args.epoch_iters

        # Evaluation settings
        self.eval_epochs = args.eval_epochs

        # Model settings
        self.feature_channel = args.feature_channel
        self.alpha = args.alpha
        self.distill = args.distill
        self.warmup_step = args.warmup_step
        self.distill_momentum = args.distill_momentum
    
        # Reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    def align(self):
        return 0




