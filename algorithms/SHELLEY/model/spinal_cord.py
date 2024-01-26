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



class SConv(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(SConv, self).__init__()

        self.in_channels = input_features
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            conv = SplineConv(input_features, output_features, dim=2, kernel_size=5, aggr="max")
            self.convs.append(conv)
            input_features = output_features

        input_features = output_features
        self.out_channels = input_features
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = [x]

        for conv in self.convs[:-1]:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]

        xs += [self.convs[-1](xs[-1], edge_index, edge_attr)]
        return xs[-1]


class SiameseSConvOnNodes(torch.nn.Module):
    def __init__(self, input_node_dim):
        super(SiameseSConvOnNodes, self).__init__()
        self.num_node_features = input_node_dim
        self.mp_network = SConv(input_features=self.num_node_features, output_features=self.num_node_features)

    def forward(self, graph):
        old_features = graph.x
        result = self.mp_network(graph)
        graph.x = old_features + 0.1 * result
        return graph


class SiameseNodeFeaturesToEdgeFeatures(torch.nn.Module):
    def __init__(self, total_num_nodes):
        super(SiameseNodeFeaturesToEdgeFeatures, self).__init__()
        self.num_edge_features = total_num_nodes

    def forward(self, graph, hyperedge=False):
        orig_graphs = graph.to_data_list()
        orig_graphs = [self.vertex_attr_to_edge_attr(graph) for graph in orig_graphs]
        if hyperedge:
            orig_graphs = [self.vertex_attr_to_hyperedge_attr(graph) for graph in orig_graphs]
        return orig_graphs

    def vertex_attr_to_edge_attr(self, graph):
        """Assigns the difference of node features to each edge"""
        flat_edges = graph.edge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)

        new_shape = (graph.edge_index.shape[1], 2, vertex_attrs.shape[1])
        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        new_edge_attrs = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        graph.edge_attr = new_edge_attrs
        return graph

    def vertex_attr_to_hyperedge_attr(self, graph):
        """Assigns the angle of node features to each hyperedge.
           graph.hyperedge_index is the incidence matrix."""
        flat_edges = graph.hyperedge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)

        new_shape = (graph.hyperedge_index.shape[1], 3, vertex_attrs.shape[1])

        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        v01 = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        v02 = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[2]
        v12 = vertex_attrs_reshaped[1] - vertex_attrs_reshaped[2]
        nv01 = torch.norm(v01, p=2, dim=-1)
        nv02 = torch.norm(v02, p=2, dim=-1)
        nv12 = torch.norm(v12, p=2, dim=-1)

        cos1 = torch.sum(v01 * v02, dim=-1) / (nv01 * nv02)
        cos2 = torch.sum(-v01 * v12, dim=-1) / (nv01 * nv12)
        cos3 = torch.sum(-v12 * -v02, dim=-1) / (nv12 * nv02)

        graph.hyperedge_attr = torch.stack((cos1, cos2, cos3), dim=-1)
        return graph