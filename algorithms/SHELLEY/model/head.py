import itertools
import numpy as np
import torch
import torch.nn as nn

from algorithms.SHELLEY.model.spinal_cord import InnerProduct , InnerProductWithWeightsAffinity
from algorithms.SHELLEY.model.loss import Distill_InfoNCE, Distill_QuadraticContrast
from algorithms.SHELLEY.utils.lap_solvers.hungarian import hungarian
from algorithms.SHELLEY.utils.lap_solvers.sinkhorn import Sinkhorn
from algorithms.SHELLEY.utils.sm_solvers.stable_marriage import stable_marriage
from algorithms.SHELLEY.utils.pad_tensor import pad_tensor

def lexico_iter(lex):
    """
    Generates lexicographically ordered pairs of elements from 
    the given collection.

    Args:
        lex (iterable): The input collection.

    Returns:
        iterable: All unique pairs of elements from the input collection.
    """
    return itertools.combinations(lex, 2)

def normalize_over_channels(x):
    """
    Normalizes the input tensor over its channels by dividing each element
     by the L2 norm of the corresponding channel.

    Args:
        x (torch.Tensor): Input tensor with multiple channels.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms

def concat_features(embeddings, num_vertices):
    """
    Concatenates embeddings along the last dimension and transposes the result.

    Args:
        embeddings (list of torch.Tensor):
            List of embeddings from different sources.
        num_vertices (list):
            List containing the number of vertices for each embedding.

    Returns:
        torch.Tensor: Concatenated and transposed tensor of embeddings.
    """
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class CommonMidbone(nn.Module):
    def __init__(self, backbone, cfg) -> None:
        """
        Args
            backbone:   The backbone model used to generate the node
                        embeddings

            cfg:        Configuration dictionary (dict of dicts) with the
                        parameter configuration.
        """
        super(CommonMidbone, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.backbone_params = backbone.parameters()
        self.vertex_affinity = InnerProduct(backbone.num_node_features)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.cfg.HEAD.SOFTMAX_TEMP))
        self.projection = nn.Sequential(
            nn.Linear(backbone.num_node_features*6, backbone.num_node_features*6, bias=True),
            nn.BatchNorm1d(backbone.num_node_features*6),
            nn.ReLU(),
            nn.Linear(backbone.num_node_features*6, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
    def forward(self, data_dict, online=True):
        # clamp temperature to be between 0.01 and 1
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052) 

        # read input dictionary
        graphs = data_dict.pyg_graphs
        n_points = data_dict.ns
        batch_size = data_dict.batch_size
        num_graphs = len(graphs)
        orig_graph_list = []

        # DEBUG
        print("[HEAD] pyg_graphs: ", graphs)

        for graph, n_p in zip(graphs, n_points):
            # DEBUG:
            print('graph:', graph)
            # extract the feature using the backbone
            if self.cfg.BACKBONE.NAME == 'gin':
                # NOTE: modified to handle batch graphs
                # node_features = self.backbone(graph)
                # apply midbone GNN
                # graph.x = node_features
                # orig_graph_list.append(graph)
                orig_graph, node_features = self.backbone(graph)
                orig_graph_list.append(orig_graph)

            else:
                raise Exception(f"Invalid backbone: {self.cfg.BACKBONE.NAME}")


        # compute the affinity matrices between source and target graphs
        unary_affs_list = [
            self.vertex_affinity([self.projection(item.x) for item in g_1], [self.projection(item.x) for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]

        # get the list of node features:
        # [[feature_g1_src, feature_g2_src, ...], [feature_g1_tgt, feature_g2_tgt, ...]]
        if self.cfg.TRAIN.LOSS_FUNC == 'distill_qc':   
            # prepare aligned node features if computing constrastive loss 
            keypoint_number_list = []   # number of keypoints in each graph pair
            node_feature_list = []      # node features for computing contrastive loss

            node_feature_graph1 = torch.zeros(
                [batch_size, data_dict['gt_perm_mat'].shape[1], node_features.shape[1]],
                device=node_features.device
            )
            node_feature_graph2 = torch.zeros(
                [batch_size, data_dict['gt_perm_mat'].shape[2], node_features.shape[1]],
                device=node_features.device
            )

            # count available keypoints in number list
            for index in range(batch_size):
                node_feature_graph1[index, :orig_graph_list[0][index].x.shape[0]] = orig_graph_list[0][index].x
                node_feature_graph2[index, :orig_graph_list[1][index].x.shape[0]] = orig_graph_list[1][index].x
                keypoint_number_list.append(torch.sum(data_dict['gt_perm_mat'][index]))
            number = int(sum(keypoint_number_list))  # calculate the number of correspondence

            # pre-align the keypoints for further computing the contrastive loss
            node_feature_graph2 = torch.bmm(data_dict['gt_perm_mat'], node_feature_graph2)
            final_node_feature_graph1 = torch.zeros([number, node_features.shape[1]], device=node_features.device)
            final_node_feature_graph2 = torch.zeros([number, node_features.shape[1]], device=node_features.device)
            count = 0
            for index in range(batch_size):
                final_node_feature_graph1[count: count + int(keypoint_number_list[index])] \
                    = node_feature_graph1[index, :int(keypoint_number_list[index])]
                final_node_feature_graph2[count: count + int(keypoint_number_list[index])] \
                    = node_feature_graph2[index, :int(keypoint_number_list[index])]
                count += int(keypoint_number_list[index])
            node_feature_list.append(self.projection(final_node_feature_graph1))
            node_feature_list.append(self.projection(final_node_feature_graph2))
        
        else:
            # TODO: if we are using another loss function
            # we don't need to prealign the node features
            raise Exception(f"[CommonMidbone] Invalid loss function: {self.cfg.MODEL.TRAIN.LOSS}")
        
        # produce output
        if online is False:
            # output of the momentum network
            return node_feature_list
        elif online is True:
            # output of the online network
            x_list = []
            for unary_affs, (idx1, idx2) in zip(unary_affs_list, lexico_iter(range(num_graphs))):
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)

                # conduct hungarian matching to get the permutation matrix for evaluation
                x = hungarian(Kp,n_points[idx1], n_points[idx2])
                x_list.append(x)
            return node_feature_list, x_list


class Common(nn.Module):
    def __init__(self, cfg, backbone):
        """
        Args:
            cfg:        Dictionary of dictionaries (easydict) that contains
                        the configuration for the main model and the backbone.
            
            backbone:   The backbone model for generating the node features.
            
        Returns:
            model:  The trained model.
        """
        super(Common, self).__init__()

        # model parameters
        self.cfg = cfg
        self.online_net = CommonMidbone(backbone, self.cfg)       # init online...
        self.momentum_net = CommonMidbone(backbone, self.cfg)     # ...and momentum network
        self.momentum = self.cfg.HEAD.MOMENTUM             # for momentum network
        self.backbone_params = list(self.online_net.backbone_params)
        self.warmup_step = self.cfg.HEAD.WARMUP_STEP       # to reach the final alpha
        self.epoch_iters = self.cfg.TRAIN.EPOCH_ITERS    

        self.model_pairs = [[self.online_net, self.momentum_net]]
        self.copy_params()  # initialize the momentum network

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # compute the distillation weight `alpha`
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = self.cfg.HEAD.ALPHA
        else:
            alpha = self.cfg.HEAD.ALPHA * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)
        
        # output of the online network
        node_feature_list, x_list = self.online_net(data_dict)

        if training is True:
            # the momentum network is only used for training
            assert self.cfg.MODEL.DISTILL is True

            # output of the momentum network
            with torch.no_grad():
                self._momentum_update()
                node_feature_m_list = self.momentum_net(data_dict, online=False)

            # loss function
            if self.cfg.TRAIN.LOSS_FUNC == 'distill_qc':
                contrastloss = Distill_InfoNCE()
                loss = contrastloss(node_feature_list, node_feature_m_list, alpha,
                                    self.online_net.logit_scale, self.momentum_net.logit_scale)
        
                crossloss = Distill_QuadraticContrast()
                loss = loss + crossloss(node_feature_list, node_feature_m_list, 
                                        self.online_net.logit_scale, self.momentum_net.logit_scale)
                
                # update data dictionary
                data_dict.update({
                    'perm_mat': x_list[0],
                    'loss': loss,
                    'ds_mat': None
                })
            
            else:
                raise Exception(f"[COMMON] Invalid loss function: {self.cfg.TRAIN.LOSS_FUNC}")
        
        else:
            # if no training,
            # directly output the result
            data_dict.update({
                'perm_mat': x_list[0],
                'ds_mat': None
            })

        return data_dict
        
    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False   # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


class StableGMMidbone(nn.Module):
    def __init__(self, backbone, cfg) -> None:
        """
        Args
            backbone:   The backbone model used to generate the node
                        embeddings

            cfg:        Configuration dictionary (dict of dicts) with the
                        parameter configuration.
        """
        super(StableGMMidbone, self).__init__()
        self.cfg = cfg
        self.backbone = backbone

        # TODO
        # ...

    def forward(self, data_dict):
        # TODO
        # ...
        return


class StableGM(nn.Module):
    def __init__(self, backbone, cfg):
        """
        Args
            backbone:   The backbone model used to generate the node
                        embeddings

            cfg:        Configuration dictionary (dict of dicts) with the
                        parameter configuration.
        """
        super(StableGM, self).__init__()
        self.cfg = cfg
        self.midbone = StableGMMidbone(backbone)

        # TODO
        # ...
    
    def forward(self, data_dict):
        # TODO
        # ...
        return