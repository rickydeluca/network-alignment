import numpy as np
import torch

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SHELLEY.backbone import GIN
from algorithms.SHELLEY.head import SIGMA


class SHELLEY(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.use_pretrained = args.use_pretrained
        self.S = None

        # backbone
        self.backbone = args.backbone
        if self.backbone == 'gin':
            self.node_feature_dim = args.node_feature_dim
            self.dim = args.dim
            self.miss_match_value = args.miss_match_value
        else:
            raise Exception(f"Invalid backbone: {self.backbone}.")

        # head
        self.head = args.head
        if self.head == 'sigma':
            self.T = args.T
            # training option
            self.lr = args.lr
            self.l2norm = args.l2norm
            self.epochs = args.epochs
            # gumbel sinkhorn option
            self.tau = args.tau
            self.n_sink_iter = args.n_sink_iter
            self.n_samples = args.n_samples
        elif self.head == 'common':
            pass
        else:
            raise Exception(f"Invalid head: {self.head}.")
        
        # device
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda is True) else 'cpu')

    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first.")
        return self.S
    
    def align(self):
        
        if self.use_pretrained:
            raise Exception("Not implemented yet.")
        
        # init backbone
        if self.backbone == 'gin':
            backbone_model = GIN(
                in_channels=self.node_feature_dim,
                out_channels=self.dim,
                dim=self.dim)
        else:
            raise Exception(f"Invalid backbone: {self.backbone}.")
        
        # init head
        if self.head == 'sigma':
            model = SIGMA(
                backbone_model,
                tau=self.tau,
                n_sink_iter=self.n_sink_iter,
                n_samples=self.n_samples).to(self.device)
            
            def pp(cost_in):
                cost = np.array(cost_in.todense())
                p = cost.sum(-1, keepdims=True)
                p = p / p.sum()
                p = torch.FloatTensor(p).to(self.device)

                cost = np.where(cost != 0)
                cost = torch.LongTensor(cost).to(self.device)

                return p, cost
            
            for gi in range(0,1):

        elif self.head == 'common':
            # TODO
            model = None
            pass
        else:
            raise Exception(f"Invalid head: {self.head}.")


