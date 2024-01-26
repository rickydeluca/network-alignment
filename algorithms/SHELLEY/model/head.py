import itertools
import numpy as np
import torch
import torch.nn as nn

from algorithms.SHELLEY.model.spinal_cord import InnerProduct, InnerProductWithWeightsAffinity
from algorithms.SHELLEY.model.backbone import get_backbone
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
            nn.Linear(backbone.num_node_features, backbone.num_node_features, bias=True),
            nn.BatchNorm1d(backbone.num_node_features),
            nn.ReLU(),
            nn.Linear(backbone.num_node_features, 256, bias=True),
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

            for graph, n_p in zip(graphs, n_points):
                # extract the feature using the backbone
                if cfg.BACKBONE.NAME == 'gin':
                    node_features = backbone(graph)
                
                    # apply midbone GNN
                    graph.x = node_features
                    orig_graph_list.append(graph)

                else:
                    raise Exception(f"Invalid backbone: {self.cfg.BACKBONE.NAME}")
                
            # compute the affinity matrices between source and target graphs
            unary_affs_list = [
                self.vertex_affinity([self.projection(item.x) for item in g_1], [self.projection(item.x) for item in g_2])
                for (g_1, g_2) in lexico_iter(orig_graph_list)
            ]

            # get the list of node features:
            # [[feature_g1_src, feature_g2_src, ...], [feature_g1_tgt, feature_g2_tgt, ...]]
            if self.cfg.MODEL.TRAIN.LOSS == 'distill_qc':   
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
        self.online_net = CommonMidbone(backbone)       # init online...
        self.momentum_net = CommonMidbone(backbone)     # ...and momentum network
        self.momentum = cfg.HEAD.MOMENTUM             # for momentum network
        self.backbone_params = list(self.online_net.backbone_params())
        self.warmup_step = cfg.HEAD.WARMUP_STEP       # to reach the final alpha
        self.epoch_iters = cfg.HEAD.TRAIN.EPOCH_ITERS    

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


class StableGM(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg:    Dictionary of dictionaries (easydict) that contains
                    the configuration for the main model and the backbone.

        Returns:
            model:  The trained model.
        """
        super(StableGM, self).__init__()

        # model parameters
        self.backbone = get_backbone(name=cfg.backbone.name, cfg=cfg.backbone)
        self.global_state_dim = cfg.model.feature_channel
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.backbone.num_node_features          # NOTE
        )
        # self.edge_affinity = InnerProductWithWeightsAffinity(
        #     self.global_state_dim,
        #     self.build_edge_features_from_node_features.num_edge_features
        # )
        self.sinkhorn = Sinkhorn(max_iter=cfg.model.sk_iter_num, epsilon=cfg.model.sk_epsilon, tau=cfg.model.sk_tau)


    def forward(self, data_dict):

        graphs = data_dict.pyg_graphs   # graphs to align
        n_points = data_dict.ns         # number of nodes in each graph
        # num_graphs = len(graphs)      # total number of graphs

        global_list = []
        orig_graph_list = []

        for graph in graphs:
            # extract feature (here we can use another backbone)
            graph = self.backbone(graph)

            # orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph = graph
            orig_graph_list.append(orig_graph)

        global_weights_list = [torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        sinkhorn_similarities = []

        for (g_1, g_2), global_weights, (ns_src, ns_tgt) in zip(lexico_iter(orig_graph_list), global_weights_list, lexico_iter(n_points)):
            similarity = self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            ns_srcs = ns_src
            ns_tgts = ns_tgt

            for i in range(len(similarity)):

                if(ns_srcs[i]==ns_tgts[i]):
                    s = self.sinkhorn(similarity[i], None, None, True)
                else:
                    s = self.sinkhorn(similarity[i], ns_srcs[i], ns_tgts[i], True)

                sinkhorn_similarities.append(s)

        # determine maximum length
        max_len_0 = max([x.size(dim=0) for x in sinkhorn_similarities])
        max_len_1 = max([x.size(dim=1) for x in sinkhorn_similarities])

        # pad all tensors to have the same length
        sinkhorn_similarities = [torch.nn.functional.pad(x, pad=(0, max_len_0 - x.size(dim=0),0,max_len_1-x.size(dim=1)), mode='constant', value=0) for x in sinkhorn_similarities]
        
        # compute the predicted permutation matrix using the stable marriage algorithm
        ss = torch.stack(sinkhorn_similarities)
        perm_mat_np = stable_marriage(ss, n_points[0], n_points[1])

        data_dict.update({
            'ds_mat': ss,
            'cs_mat': ss,
            'perm_mat':perm_mat_np,
        })
        
        return data_dict


def get_head(name, cfg):
    if name == 'common':
        model = Common(cfg)
    elif name == 'stablegm':
        model = StableGM(cfg)
    else:
        raise Exception(f"Invalid head: {name}.")
    
    return model