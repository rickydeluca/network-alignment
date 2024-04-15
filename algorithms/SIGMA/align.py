import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected, to_scipy_sparse_matrix
from torch_geometric.utils.convert import from_networkx

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SIGMA.gnns import GIN
from algorithms.SIGMA.model import SIGMA
from algorithms.SIGMA.train import train
from utils.graph_utils import load_gt


class SIGMA_Aligner(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, train_dict, cfg):
        """
        Initialize SIGMA.

        Args:
            source_dataset:     The source network dataset.

            target_dataset:     The target network dataset.

            train_dict:         Dictionary with source-target alignmnets
                                for training.

            cfg:                The model configuration dictionary.
        """

        super(SIGMA_Aligner, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.train_dict = train_dict
        self.cfg = cfg

        # training groundtruth
        gt = load_gt(self.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.gt_train = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k,v in gt.items()}

        # alignment
        self.S = None
        self.source_embedding = None
        self.target_embedding = None
        self.source_train_nodes = np.array(list(self.gt_train.keys()))

        # device
        self.device = torch.device(self.cfg.DEVICE)

    def get_alignment_matrix(self):
        return self.S

    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding
    
    def align(self):
        # construct model
        f_update = GIN(in_channels=self.cfg.GIN.NODE_FEATURE_DIM,
                       out_channels=self.cfg.GIN.DIM, 
                       dim=self.cfg.GIN.DIM)
        model = SIGMA(f_update,
                      tau=self.cfg.SIGMA.TAU,
                      n_sink_iter=self.cfg.SIGMA.N_SINK_ITER,
                      n_samples=self.cfg.SIGMA.N_SAMPLES).to(self.device)
        
        # convert networkx to pyg
        edge_index_s = to_scipy_sparse_matrix(from_networkx(self.source_dataset.G).edge_index)
        edge_index_t = to_scipy_sparse_matrix(from_networkx(self.target_dataset.G).edge_index)

        idx2node_s = {idx: node for node, idx in self.source_dataset.id2idx.items()}
        idx2node_t = {idx: node for node, idx in self.target_dataset.id2idx.items()}
        num_nodes = min([len(idx2node_s), len(idx2node_t)])

        p_s, cost_s = self.pp(edge_index_s)
        p_t, cost_t = self.pp(edge_index_t)

        # create pyg graphs
        graph_s = Data(x=p_s, edge_index=cost_s)
        graph_t = Data(x=p_t, edge_index=cost_t)
        
        # train
        model = train(model, graph_s, graph_t, idx2node_s, idx2node_t, num_nodes, self.cfg)

        # align
        model.eval()
        logits_t, _ = model(p_s, cost_s, p_t, cost_t, self.cfg.SIGMA.T, miss_match_value=self.cfg.SIGMA.MISS_MATCH_VALUE)
        self.S = logits_t.detach().cpu().numpy()[:58, :58]
        print('S', self.S.shape)
        print(self.S)
        return self.S
    
    def pp(self, cost_in):
        cost = np.array(cost_in.todense())
        p = cost.sum(-1, keepdims=True)
        p = p / p.sum()
        p = torch.FloatTensor(p).to(self.device)

        cost = np.where(cost != 0)
        cost = torch.LongTensor(cost).to(self.device)

        return p, cost

