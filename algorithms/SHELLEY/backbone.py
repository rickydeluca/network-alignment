import numpy as np
import torch
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GINConv


class GIN(torch.nn.Module):
    """
    The graph nerual network described in "Stochastic Iterative Graph Matching"
    by Liu et al. (2021).
    """
    def __init__(self, in_channels, out_channels, dim, softmax_temp=1, bias=True):
        super(GIN, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / softmax_temp))

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

    def forward(self, data_dict):
        # read data from input dictionary
        graphs = [data_dict['source_graph'], data_dict['target_graph']]

        embedding_list = []
        for graph in graphs:
            X = graph.x
            X_importance = graph.x_importance
            edge_index = graph.edge_index

            # DEBUG
            # res = X * X_importance
            # print('X_shape', X.shape)
            # print('X', X)
            # print('X_importance', X_importance.shape)
            # print('X * X_importace', res.shape)

            # forward step
            x = self.bn_in(X * X_importance)  # DEBUG
            xs = [x]

            xs.append(self.conv1(xs[-1], edge_index))
            xs.append(self.conv2(xs[-1], edge_index))
            xs.append(self.conv3(xs[-1], edge_index))
            xs.append(self.conv4(xs[-1], edge_index))
            xs.append(self.conv5(xs[-1], edge_index))
            xs.append(torch.tanh(self.fc1(xs[-1])))

            x = torch.cat(xs[1:], dim=-1)

            embedding_list.append(x)

        return embedding_list
    

def get_backbone(name: str, cfg: dict):
    """
    Return the backbone defined by `name` initialized
    with the arguments in `args` dictionary.
    """

    if name == 'gin':
        assert 'node_feature_dim' in cfg
        assert 'dim' in cfg
        
        in_channels = cfg.node_feature_dim
        out_channels = cfg.dim 
        dim = cfg.dim
        softmax_temp = cfg.softmax_temp

        return GIN(in_channels, out_channels, dim, softmax_temp=softmax_temp, bias=True)

    else:
        raise Exception(f"Invalid backbone name: {name}.")