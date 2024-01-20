import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, feature_channel=None, softmax_temp=None, rescale=None):
        super(SplineCNN, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=feature_channel * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.vertex_affinity = InnerProduct(256)
        self.rescale = rescale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / softmax_temp))

        self.projection = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self, data_dict, online=True):
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)  # Clamp temperature to be between 0.01 and 1
        
        source_graph = data_dict['source_graph']
        target_graph = data_dict['target_graph']
        
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
    def __init__(self, source_embedding, target_embedding, backbone=None,
                 feature_channel=None, softmax_temp=None, momentum=None,
                 warmup_step=None, epoch_iters=None, rescale=None, alpha=None):

        super(CommonMapping, self).__init__()
        self.source_embedding = source_embedding
        self.target_embeddings = target_embedding
        self.loss_fn = MappingLossFunctions()

        if backbone == 'splinecnn':
            self.online_net = SplineCNN(
                feature_channel=feature_channel,
                softmax_temp=softmax_temp,
                rescale=rescale)
            self.momentum_net = SplineCNN(
                feature_channel=feature_channel,
                softmax_temp=softmax_temp,
                rescale=rescale)
        else:
            raise ValueError(f"Backbone '{backbone}' is not implemented!")

        self.backbone_params = None     # TODO
        self.momentum = momentum
        self.warmup_step = warmup_step
        self.epoch_iters = epoch_iters
        self.alpha = alpha

        self.model_pairs = [[self.online_net, self.momentum_net]]
        self.copy_params()  # Init momentum network

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # Compute distillation weight `alpha`
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = self.alpha
        else:
            alpha = self.alpha * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)

        # Get output from online network
        online_embedding, perm_mat = self.online_net(data_dict, online=True)

        if training:
            # Get output from momentum network
            with torch.no_grad():
                self._momentum_update()
                momentum_embedding = self.momentum_net(data_dict, online=False)

            # Compute loss
            loss = self.loss_fn.loss(feature=online_embedding,
                                     feature_m=momentum_embedding,
                                     alpha=alpha,
                                     dynamic_temperature=self.online_net.logit_scale,
                                     dynamic_temperature_m=self.momentum_net.logit_scale,
                                     groundtruth=data_dict['gt_perm_mat'])

            # Update dictionary
            data_dict.update({
                'perm_mat': perm_mat,
                'loss': loss,
                'ds_mat': None
            })

        else:   # Directly output the result
            data_dict.update({
                'perm_mat': perm_mat,
                'ds_mat': None
            })

        return data_dict

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # Initialize
                param_m.requires_grad = False   # Not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)