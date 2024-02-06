import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv


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


class InnerProductWithWeightsAffinity(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InnerProductWithWeightsAffinity, self).__init__()
        self.d = output_dim
        self.A = torch.nn.Linear(input_dim, output_dim)

    def _forward(self, X, Y, weights):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        coefficients = torch.tanh(self.A(weights))
        res = torch.matmul(X * coefficients, Y.transpose(0, 1))
        res = torch.nn.functional.softplus(res)
        return res

    def forward(self, Xs, Ys, Ws):
        return [self._forward(X, Y, W) for X, Y, W in zip(Xs, Ys, Ws)]