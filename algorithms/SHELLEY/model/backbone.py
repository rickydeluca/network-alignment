import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GINConv

from algorithms.PALE.loss import EmbeddingLossFunctions


def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled



class PaleEmbedding(torch.nn.Module):
    def __init__(self,
                 n_nodes:int = None,
                 embedding_dim:int = None,
                 deg:np.ndarray = None,
                 neg_sample_size:int = None,
                 cuda:bool = False):

        """
        Parameters
        ----------
        n_nodes: int
            Number of all nodes
        embedding_dim: int
            Embedding dim of nodes
        deg: ndarray , shape = (-1,)
            Array of degrees of all nodes
        neg_sample_size : int
            Number of negative candidate to sample
        cuda: bool
            Whether to use cuda
        """

        super(PaleEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.deg = deg
        self.neg_sample_size = neg_sample_size
        self.link_pred_layer = EmbeddingLossFunctions()
        self.n_nodes = n_nodes
        self.use_cuda = cuda


    def loss(self, nodes, neighbor_nodes):
        batch_output, neighbor_output, neg_output = self.forward(nodes, neighbor_nodes)
        batch_size = batch_output.shape[0]
        loss, loss0, loss1 = self.link_pred_layer.loss(batch_output, neighbor_output, neg_output)
        loss = loss/batch_size
        loss0 = loss0/batch_size
        loss1 = loss1/batch_size
        
        return loss, loss0, loss1


    def forward(self, nodes, neighbor_nodes=None):
        node_output = self.node_embedding(nodes)
        node_output = F.normalize(node_output, dim=1)

        if neighbor_nodes is not None:
            neg = fixed_unigram_candidate_sampler(
                num_sampled=self.neg_sample_size,
                unique=False,
                range_max=len(self.deg),
                distortion=0.75,
                unigrams=self.deg
                )

            neg = torch.LongTensor(neg)
            
            if self.use_cuda:
                neg = neg.cuda()
            neighbor_output = self.node_embedding(neighbor_nodes)
            neg_output = self.node_embedding(neg)
            # normalize
            neighbor_output = F.normalize(neighbor_output, dim=1)
            neg_output = F.normalize(neg_output, dim=1)

            return node_output, neighbor_output, neg_output

        return node_output

    def get_embedding(self):
        nodes = np.arange(self.n_nodes)
        nodes = torch.LongTensor(nodes)
        if self.use_cuda:
            nodes = nodes.cuda()
        embedding = None
        BATCH_SIZE = 512
        for i in range(0, self.n_nodes, BATCH_SIZE):
            j = min(i + BATCH_SIZE, self.n_nodes)
            batch_nodes = nodes[i:j]
            if batch_nodes.shape[0] == 0: break
            batch_node_embeddings = self.forward(batch_nodes)
            if embedding is None:
                embedding = batch_node_embeddings
            else:
                embedding = torch.cat((embedding, batch_node_embeddings))

        return embedding



class GIN(torch.nn.Module):
    """
    The graph nerual network described in "Stochastic Iterative Graph Matching"
    by Liu et al. (2021).
    """
    def __init__(self, in_channels, out_channels, dim, bias=True):
        super(GIN, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False
        self.num_node_features = out_channels
        self.out_dimension = out_channels*6

        self.bn_in = torch.nn.BatchNorm1d(in_channels)

        self.nn1 = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv1 = GINConv(self.nn1, eps=eps, train_eps=train_eps)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.nn2 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv2 = GINConv(self.nn2, eps=eps, train_eps=train_eps)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.nn3 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv3 = GINConv(self.nn3, eps=eps, train_eps=train_eps)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.nn4 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv4 = GINConv(self.nn4, eps=eps, train_eps=train_eps)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.nn5 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv5 = GINConv(self.nn5, eps=eps, train_eps=train_eps)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, out_channels, bias=False)


    def forward(self, graph):
        X = graph.x.to(torch.float)
        X_importance = graph.x_importance
        edge_index = graph.edge_index

        # Forward step
        x = self.bn_in(X * X_importance)  # NOTE
        xs = [x]

        xs.append(self.conv1(xs[-1], edge_index))
        xs.append(self.conv2(xs[-1], edge_index))
        xs.append(self.conv3(xs[-1], edge_index))
        xs.append(self.conv4(xs[-1], edge_index))
        xs.append(self.conv5(xs[-1], edge_index))
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x
    

def get_backbone(name: str, cfg: dict):
    """
    Return the backbone defined by `name` initialized
    with the arguments in `args` dictionary.
    """

    if name == 'gin':
        assert 'node_feature_dim' in cfg
        assert 'dim' in cfg
        assert 'softmax_temp' in cfg
        
        in_channels = cfg.node_feature_dim
        out_channels = cfg.dim 
        dim = cfg.dim
        softmax_temp = cfg.softmax_temp

        return GIN(in_channels, out_channels, dim, softmax_temp=softmax_temp, bias=True)

    else:
        raise Exception(f"Invalid backbone name: {name}.")