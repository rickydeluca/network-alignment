import torch
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GINConv


class GIN(torch.nn.Module):
    """
    The graph nerual network described in "Stochastic Iterative Graph Matching"
    by Liu et al. (2021).
    """
    def __init__(self, in_channels, out_channels, dim, bias=True):
        super(GIN, self).__init__()

        '''
        in_channels = cfg.in_channels
        out_channels = cfg.out_channels
        dim = cfg.dim
        bias = cfg.bias
        '''

        act = Tanh()
        eps = 0.0
        train_eps = False

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

    def forward(self, data_dict, online=True):
        # read data from input dictionary
        X = data_dict.X
        X_importance = data_dict.X_importance
        edge_index = data_dict.edge_index

        # forward step
        x = self.bn_in(X * X_importance)
        xs = [x]

        xs.append(self.conv1(xs[-1], edge_index))
        xs.append(self.conv2(xs[-1], edge_index))
        xs.append(self.conv3(xs[-1], edge_index))
        xs.append(self.conv4(xs[-1], edge_index))
        xs.append(self.conv5(xs[-1], edge_index))
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x
    

def get_backbone(name: str, args: dict):
    """
    Return the backbone defined by `name` initialized
    with the arguments in `args` dictionary.
    """

    if name == 'gin':
        assert 'in_channels' in args
        assert 'out_channels' in args
        assert 'dim' in args
        assert 'bias' in args

        return GIN(args)

    else:
        raise Exception(f"Invalid backbone name: {name}.")