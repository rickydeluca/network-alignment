import itertools

import torch
import torch.nn as nn

from algorithms.SHELLEY.model.spinal_cord import InnerProduct, InnerProductWithWeightsAffinity
from algorithms.SHELLEY.model.backbone import get_backbone
from algorithms.SHELLEY.model.loss import get_loss_function
from algorithms.SHELLEY.utils.lap_solvers.hungarian import hungarian
from algorithms.SHELLEY.utils.lap_solvers.sinkhorn import Sinkhorn
from algorithms.SHELLEY.utils.sm_solvers.stable_marriage import stable_marriage

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



class Common(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg:    Dictionary of dictionaries (easydict) that contains
                    the configuration for the main model and the backbone.
            
        Returns:
            model:  The trained model.
        """
        super(Common, self).__init__()

        # model parameters
        self.online_net = get_backbone(name=cfg.backbone.name, cfg=cfg.backbone)
        self.momentum_net = get_backbone(name=cfg.backbone.name, cfg=cfg.backbone)
        self.momentum = cfg.model.momentum          # for momentum network
        self.warmup_step = cfg.model.warmup_step    # to reach the final alpha
        self.epoch_iters = cfg.model.epoch_iters    
        self.final_alpha = cfg.model.alpha          # target after warmup step
        self.distill = cfg.model.distill            # boolean

        self.backbone_params = list(self.online_net.parameters())   # NOTE

        self.model_pairs = [[self.online_net, self.momentum_net]]
        self.copy_params()  # init momentum network

        self.vertex_affinity = InnerProduct(self.online_net.num_node_features)   # NOTE
        self.projection = nn.Sequential(
            nn.Linear(self.online_net.num_node_features, self.online_net.num_node_features, bias=True),
            nn.BatchNorm1d(self.online_net.num_node_features),
            nn.ReLU(),
            nn.Linear(self.online_net.num_node_features, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # define loss function
        self.loss_name = cfg.model.train.loss
        self.loss_fn = get_loss_function(name=self.loss_name)

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # compute the distillation weight `alpha`
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = self.final_alpha
        else:
            alpha = self.final_alpha * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)
        
        # read input
        graphs = data_dict.pyg_graphs
        n_points = data_dict.ns
        batch_size = data_dict['ns']
        num_graphs = len(graphs)

        orig_graph_list = []

        for graph in graphs:
            graph = self.online_net(graph)
            orig_graph = graph
            orig_graph_list.append(orig_graph)
        
        # compute node affinity between graphs
        unary_affs_list = [
            self.vertex_affinity([self.projection(item.x) for item in g_1], [self.projection(item.x) for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]

        # get output of the online network
        embedding_src = self.online_net(data_dict.source_graph) # out: [source_emb_tensor, target_emb_tensor]
        embedding_tgt = self.online_net(data_dict.target_graph)
        embedding_list = [embedding_src, embedding_tgt]

        # generate node affinity matrix using the node embeddings
        aff_mat = self.vertex_affinity(embedding_list[0], embedding_list[1])

        # generate the permutation matrix (the predicted alignments)
        num_source_nodes = torch.tensor(aff_mat.shape[0]).unsqueeze(0)
        num_target_nodes = torch.tensor(aff_mat.shape[1]).unsqueeze(0)
        perm_mat = hungarian(aff_mat, num_source_nodes, num_target_nodes)

        if training is True:
            # the momentum network is used only during training
            assert self.distill

            # get output of the momentum network
            with torch.no_grad():
                self._momentum_update()
                embedding_m_src = self.momentum_net(data_dict.source_graph)
                embedding_m_tgt = self.momentum_net(data_dict.target_graph)
                embedding_m_list = [embedding_m_src, embedding_m_tgt]

            # compute loss
            if self.loss_name == 'common':
                embedding_list_batch = [embedding_list[0][data_dict.source_batch], embedding_list[1][data_dict.target_batch]]
                embedding_m_list_batch = [embedding_m_list[0][data_dict.source_batch], embedding_m_list[1][data_dict.target_batch]]
            
                loss = self.loss_fn(feature=embedding_list_batch,
                                    feature_m=embedding_m_list_batch,
                                    alpha=alpha,
                                    dynamic_temperature=self.online_net.logit_scale,
                                    dynamic_temperature_m=self.momentum_net.logit_scale,
                                    groundtruth=data_dict.gt_perm_mat)
            else:
                raise Exception(f"Invalid loss function: {self.loss_name}")


            # update the dictionary
            data_dict.update({
                'aff_mat': aff_mat,
                'perm_mat': perm_mat,
                'loss': loss,
                'ds_mat': None
            })

        else:  # directly output the predicted alignment
            data_dict.update({
                'aff_mat': aff_mat,
                'perm_mat': perm_mat,
                'ds_mat': None
            })

        return data_dict

        
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