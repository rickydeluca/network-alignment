import itertools
import numpy as np
import torch
import torch.nn as nn

from algorithms.SHELLEY.model.affinity_layer import InnerProduct, InnerProductWithWeightsAffinity
from algorithms.SHELLEY.model.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
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
        
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.HEAD.FEATURE_CHANNEL * 3)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.vertex_affinity = InnerProduct(backbone.num_node_features)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.cfg.HEAD.SOFTMAX_TEMP))
        self.projection = nn.Sequential(
            nn.Linear(backbone.out_dimension, backbone.out_dimension, bias=True),
            nn.BatchNorm1d(backbone.out_dimension),
            nn.ReLU(),
            nn.Linear(backbone.out_dimension, backbone.num_node_features, bias=True),
            nn.BatchNorm1d(backbone.num_node_features),
            nn.ReLU()
        )
        
    def forward(self, data_dict, online=True):
        # Clamp temperature to be between 0.01 and 1
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052) 

        # Read input dictionary
        graphs = data_dict['pyg_graphs']
        n_points = data_dict['ns']
        batch_size = data_dict['batch_size']
        num_graphs = data_dict['num_graphs']
        orig_graph_list = []

        for graph, n_p in zip(graphs, n_points):
            # Extract the feature using the backbone
            if self.cfg.BACKBONE.NAME in ['gin', 'pale_linear']:
                node_features = self.backbone(graph).to(self.cfg.HEAD.DEVICE)
            else:
                raise Exception(f"Invalid backbone: {self.cfg.BACKBONE.NAME}")
            
            # GNN
            graph.x = node_features
            graph = self.message_pass_node_features(graph.to(self.cfg.HEAD.DEVICE))
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        # Compute the affinity matrices between source and target graphs
        unary_affs_list = [
            self.vertex_affinity([self.projection(item.x) for item in g_1], [self.projection(item.x) for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]

        if self.cfg.TRAIN.LOSS_FUNC == 'distill_qc':   
            # Prepare aligned node features if computing constrastive loss 
            keypoint_number_list = []   # Number of keypoints in each graph pair
            node_feature_list = []      # Node features for computing contrastive loss

            node_feature_graph1 = torch.zeros(
                [batch_size, data_dict['gt_perm_mat'].shape[1], node_features.shape[1]],
                device=node_features.device
            )
            node_feature_graph2 = torch.zeros(
                [batch_size, data_dict['gt_perm_mat'].shape[2], node_features.shape[1]],
                device=node_features.device
            )

            # Count available keypoints in number list
            for index in range(batch_size):
                node_feature_graph1[index, :orig_graph_list[0][index].x.shape[0]] = orig_graph_list[0][index].x
                node_feature_graph2[index, :orig_graph_list[1][index].x.shape[0]] = orig_graph_list[1][index].x
                keypoint_number_list.append(torch.sum(data_dict['gt_perm_mat'][index]))
            number = int(sum(keypoint_number_list))  # Calculate the number of correspondence

            # Pre-align the keypoints for further computing the contrastive loss
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
        
        # Produce output
        if online is False:
            # Output of the momentum network
            return node_feature_list
        elif online is True:
            # Output of the online network
            x_list = []
            for unary_affs, (idx1, idx2) in zip(unary_affs_list, lexico_iter(range(num_graphs))):
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)

                # Conduct hungarian matching to get the permutation matrix for evaluation
                x = hungarian(Kp, n_points[idx1], n_points[idx2])
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

        # Model parameters
        self.cfg = cfg
        self.online_net = CommonMidbone(backbone, self.cfg)             # Init online...
        self.momentum_net = CommonMidbone(backbone, self.cfg)           # ...and momentum network
        self.momentum = self.cfg.HEAD.MOMENTUM                          # For momentum network
        self.backbone_params = list(self.online_net.backbone_params)    # Used if case of separate lr
        self.warmup_step = self.cfg.HEAD.WARMUP_STEP                    # To reach the final alpha
        self.epoch_iters = self.cfg.TRAIN.EPOCH_ITERS

        self.model_pairs = [[self.online_net, self.momentum_net]]
        self.copy_params()  # Initialize the momentum network

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # Compute the distillation weight `alpha`
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = self.cfg.HEAD.ALPHA
        else:
            alpha = self.cfg.HEAD.ALPHA * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)
        
        # Output of the online network
        node_feature_list, x_list = self.online_net(data_dict)

        if training is True:
            # The momentum network is only used for training
            assert self.cfg.HEAD.DISTILL is True

            # Output of the momentum network
            with torch.no_grad():
                self._momentum_update()
                node_feature_m_list = self.momentum_net(data_dict, online=False)

            # Loss function
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
            # If no training, directly output the result
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


class _StableGMMidbone(nn.Module):
    def __init__(self, backbone, cfg) -> None:
        """
        Args
            backbone:   The backbone model used to generate the node
                        embeddings

            cfg:        Configuration dictionary (dict of dicts) with the
                        parameter configuration.
        """
        super(_StableGMMidbone, self).__init__()
        self.cfg = cfg
        self.backbone = backbone

        # TODO
        # ...

    def forward(self, data_dict):
        # TODO
        # ...
        return


class _StableGM(nn.Module):
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
        self.midbone = _StableGMMidbone(backbone)

        # TODO
        # ...
    
    def forward(self, data_dict):
        # TODO
        # ...
        return
    

class StableGM(torch.nn.Module):
    def __init__(self, backbone, cfg):
        super(StableGM, self).__init__()
        self.cfg = cfg              # Configuration dicrionary
        self.backbone = backbone    # Backbone model
        self.global_state_dim = self.cfg.HEAD.FEATURE_CHANNEL
        self.vertex_affinity = InnerProductWithWeightsAffinity(self.global_state_dim, self.backbone.num_node_features)
        self.sinkhorn = Sinkhorn(max_iter=self.cfg.HEAD.SK_ITER_NUM, epsilon=self.cfg.HEAD.SK_EPSILON, tau=self.cfg.HEAD.SK_TAU)

    def forward(
        self,
        data_dict,
    ):
        # Read input dictionary
        graphs = data_dict['pyg_graphs']
        n_points = data_dict['ns']
        batch_size = data_dict['batch_size']
        num_graphs = data_dict['num_graphs']
        

        if 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        else:
            raise ValueError('Ground truth information is required during training.')

        global_list = []
        orig_graph_list = []
        for n_p, _graph in zip(n_points, graphs):
            # Generate node features using the backbone
            graph = _graph.clone()
            node_features = self.backbone(graph)
            graph.x = node_features

            # Unroll the graph in the batch
            orig_graphs = graph.to_data_list()
            orig_graph_list.append(orig_graphs)
            
       
        global_weights_list = [torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        sinkhorn_similarities = []

        for (g_1, g_2), global_weights,(ns_src, ns_tgt) in zip(lexico_iter(orig_graph_list), global_weights_list,lexico_iter(n_points)):
            similarity=self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            ns_srcs = ns_src
            ns_tgts = ns_tgt
            
            for i in range(len(similarity)):
                if(ns_srcs[i]==ns_tgts[i]):
                    s = self.sinkhorn(similarity[i], None, None,True)
                else:
                    s = self.sinkhorn(similarity[i], ns_srcs[i], ns_tgts[i],True)
                sinkhorn_similarities.append(s)
                    
        # Determine maximum length
        max_len_0 = max([x.size(dim=0) for x in sinkhorn_similarities])
        max_len_1= max([x.size(dim=1) for x in sinkhorn_similarities])
        
        # Pad all tensors to have same length
        sinkhorn_similarities = [torch.nn.functional.pad(x, pad=(0, max_len_0 - x.size(dim=0),0,max_len_1-x.size(dim=1)), mode='constant', value=0) for x in sinkhorn_similarities]
        ss = torch.stack(sinkhorn_similarities)
    
        perm_mat_np =stable_marriage(ss, n_points[0], n_points[1])
        data_dict.update({
            'ds_mat': ss,
            'cs_mat': ss,
            'perm_mat':perm_mat_np,
        })
        
        return data_dict