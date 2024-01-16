import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE

from algorithms.SANE.loss import EmbeddingLossFunctions

class SaneEmbedding(nn.Module):
    def __init__(self, source_data, target_data, neg_sample_ratio=2.0):

        """
        Args:
        - n_nodes (int):            Number of all nodes
        - embedding_dim (int):      Node embedding size
        - neg_sample_size (int):    Number of negative candidate to sample
        """

        super(SaneEmbedding, self).__init__()
        self.source_data = source_data
        self.target_data = target_data
        self.neg_sample_ratio = neg_sample_ratio
        self.loss_fn = EmbeddingLossFunctions()

class SaneEmbeddingGCN(SaneEmbedding):
    """
    Class to handle a spectral-based graph convolutional network.
    It takes in input two graphs expressed in pytorch geometric data format
    with the corresponding node feature matrices and perform a convolutional
    operation on both networks.
    """
    def __init__(self, source_data, target_data, neg_sampling_ratio=2, in_channels=64, hidden_channels=64, out_channels=64, num_hidden_layers=0):
        
        super(SaneEmbeddingGCN).__init__(source_data, target_data, neg_sample_ratio=neg_sampling_ratio)
        
        if isinstance(in_channels, int):
            in_channels = (in_channels)

        assert(len(in_channels) == num_hidden_layers)
        
        if num_hidden_layers == 0:
            self.convs = nn.ModuleList([GCNConv(in_channels, out_channels)])
        elif num_hidden_layers >= 1:
            self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels[0])])
            for i in range(1, num_hidden_layers):
                self.convs.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))
            self.convs.append(GCNConv(hidden_channels[num_hidden_layers-1], out_channels))
        else:
            raise ValueError("The number of hidden layers must be greater or equal 0.")


    def loss(self, source_indices, target_indices):
        pass
        
    

class GCNEmbeddingModel():
    """
    Class to handle a spectral-based graph convolutional network.
    It takes in input two graphs expressed in pytorch geometric data format
    with the corresponding node feature matrices and perform a convolutional
    operation on both networks.
    """
    def __init__(self, in_channels=64, hidden_channels=(64), out_channels=64, num_hidden_layers=0):
        super().__init__()
        if num_hidden_layers == 0:
            self.convs = nn.ModuleList([GCNConv(in_channels, out_channels)])
        elif num_hidden_layers >= 1:
            self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels[0])])
            for i in range(1, num_hidden_layers):
                self.convs.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))
            self.convs.append(GCNConv(hidden_channels[num_hidden_layers-1], out_channels))
        else:
            raise ValueError("The number of hidden layers must be greater or equal 0.")

    def forward(self, source_data, target_data):
        edge_index1 = source_data.edge_index
        edge_index2 = target_data.edge_index

        if 'weight' in source_data.keys() and 'weight' in target_data.keys():
            edge_weight1 = source_data.weight
            edge_weight2 = target_data.weight
        else:
            edge_weight1 = None
            edge_weight2 = None
            
        x1 = source_data.x
        x2 = target_data.x

        for conv in self.convs:
            # Source network.
            x1 = conv(x1, edge_index1, edge_weight1)
            x1 = F.relu(x1)

            # Target network.
            x2 = conv(x2, edge_index2, edge_weight2)
            x2 = F.relu(x2)

        return x1, x2
    
class SaneEmbeddingSAGE(nn.Module):
    """
    Class to handle an embedding model based on GraphSAGE.
    It takes in input two graphs expressed in pytorch geometric data format
    with the corresponding attributes and run GraphSAGE on them to generate
    the node embeddings for both graphs.
    """

    def __init__(self, in_channels=64, hidden_channels=(64), out_channels=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.conv = GraphSAGE(in_channels=in_channels,
                              hidden_channels=hidden_channels,
                              out_channels=out_channels,
                              num_layers=num_layers,
                              dropout=dropout)
        
    def loss(self, source_indices, target_indices):
        pass

        
    def forward(self, source_data, target_data):
        edge_index1 = source_data.edge_index
        edge_index2 = target_data.edge_index

        if 'weight' in source_data.keys() and 'weight' in target_data.keys():
            edge_weight1 = source_data.weight
            edge_weight2 = target_data.weight
        else:
            edge_weight1 = None
            edge_weight2 = None
            
        x1 = source_data.x
        x2 = target_data.x

        # SAGE Conv
        x1 = self.conv(x1, edge_index1)
        x2 = self.conv(x2, edge_index2)

        return x1, x2