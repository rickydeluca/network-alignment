import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.COMMON.lap_solvers import hungarian
from algorithms.COMMON.loss import MappingLossFunctions
from algorithms.COMMON.sconv_archs import (
    SiameseNodeFeaturesToEdgeFeatures,
    SiameseSConvOnNodes,
)


class InnerProduct(nn.Module):
    def __init__(self, output_dim):
        super(InnerProduct, self).__init__()
        self.d = output_dim

    def _forward(self, X, Y):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        X = torch.nn.functional.normalize(X, dim=-1)
        Y = torch.nn.functional.normalize(Y, dim=-1)
        res = torch.matmul(X, Y.transpose(0, 1))
        return res

    def forward(self, Xs, Ys):
        return [self._forward(X, Y) for X, Y in zip(Xs, Ys)]


class SplineCNN(nn.Module):
    def __init__(self):
        super(SplineCNN, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.COMMON.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.vertex_affinity = InnerProduct(256)
        self.rescale = cfg.PROBLEM.RESCALE
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.COMMON.SOFTMAXTEMP))

        self.projection = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self, source_graph, target_graph, online=True):
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)  # clamp temperature to be between 0.01 and 1
        

        # Apply Spline Convolution
        conv_source_graph = self.message_pass_node_features(source_graph)
        conv_source_graph = self.build_edge_features_from_node_features(conv_source_graph)
        
        conv_target_graph = self.message_pass_node_features(target_graph)
        conv_target_graph = self.build_edge_features_from_node_features(conv_target_graph)
        
        # Compute vertex affinity
        conv_source_embedding = self.projection(conv_source_graph.x)
        conv_target_embedding = self.projection(conv_target_graph.x)

        unary_affinity = self.vertex_affinity(conv_source_embedding,
                                                conv_target_embedding)
        
        embedding_list = [conv_source_embedding, conv_target_embedding]

        if online:
            Kp = torch.tensor(unary_affinity)

            # Get permutation matrix through the Hungarian matching algorithm
            num_source_nodes = conv_source_graph.num_nodes
            num_target_nodes = conv_target_graph.num_nodes
            perm_matrix = hungarian(Kp, num_source_nodes, num_target_nodes)

            return embedding_list, perm_matrix
        else:
            # Output of the momentum network
            return embedding_list
        



class CommonMapping(nn.Module):
    def __init__(self, source_embedding, target_embedding, backbone='splicecnn'):

        super(CommonMapping, self).__init__()
        self.source_embedding = source_embedding
        self.target_embeddings = target_embedding
        self.loss_fn = MappingLossFunctions()

        if backbone == 'splinecnn':
            self.online_net = SplineCNN()
            self.momentum_net = SplineCNN()
        else:
            raise ValueError(f"Bacbone '{backbone}' not implemented!")
