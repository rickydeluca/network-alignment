import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx

from algorithms.HDA.hyperdimensional_encoding import *
from algorithms.SANE.embedding_model import *
from algorithms.SANE.mapping_model import *


class PairData(Data):
    """
    Class to handle a pair of Data objects
    to perform graph matching.
    """
    def __inc__(self, key, value, *args, **kwargs):
        """
        Increase the indices of source and target networks wrt their
        corresponding number of nodes.
        """
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def dict_to_tensor(dictionary):
    return torch.tensor(list(dictionary.values()), dtype=torch.float32)


def networkx_to_pyg(dataset):
    """
    Convert a NetworkX graph in a edge-list representation 
    useful for PyG.
    """
    # Get network informations from the dataset
    G = dataset.G
    id2idx = dataset.id2idx
    idx2id = {idx: id for id, idx in id2idx.items()}
    node_feats = torch.tensor(dataset.features, dtype=torch.float32)

    if dataset.edge_features is not None:
        edge_feats = torch.tensor(dataset.edge_features)

    # Generate PyG representation
    data = from_networkx(G)
    data.x = node_feats

    if dataset.edge_features is not None:
        data.edge_attr = edge_feats

    return data #, idx2id


def read_network_dataset(dataset, pos_info=False):

    # Convert NetworX to Data representation
    G = from_networkx(dataset.G)

    # Extract useful informations
    edge_index = G.edge_index
    edge_weight = G.weight if 'weight' in G.keys() else None

    # Read other features    
    x = torch.tensor(dataset.features, dtype=torch.float32) if dataset.features is not None else None
    edge_attr = torch.tensor(dataset.edge_features) if dataset.edge_features is not None else None

    # Use also positional informations if required
    if pos_info:
        adj_matrix = torch.tensor(nx.adjacency_matrix(dataset.G).todense(), dtype=torch.float32)
        if x is not None:
            x = torch.cat((x, adj_matrix), dim=1)
    
    # Display informations
    # print("edge_index:", edge_index.shape)
    # print("edge_weight:", edge_weight.shape if edge_weight is not None else None)
    # print("x:", x.shape)
    # print("edge_attr:", edge_attr)

    # Build the Data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

    return data


def get_mapping_model(model, input_dim=0, dropout=0.5):
    if model == 'cosine_similarity':
        predictor = CosineSimilarity()
    elif model == 'inner_product':
        predictor = InnerProduct()
    elif model == 'dnn':
        predictor = LinkPredictor(input_dim=input_dim, dropout=dropout)
    else:
        raise ValueError(f"{model} is an invalid model name. Choose from: 'cosine_similarity', 'inner_product', 'dnn'")

    return predictor


def get_embedding_model(model, in_channels=None, hidden_channels=None, out_channels=None, num_layers=None, heads=8, concat=False, dropout=0.0):

    if model == 'sage':
        embedder = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout
        )
    elif model == 'spectral':
        embedder = SpectralGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_hidden_layers=num_layers-1
        )
    elif model == 'cross_transform':
        embedder = CrossTransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            heads=heads,
            concat=concat
        )
    else:
        raise ValueError(f"{model} is an invalid model name. Choose from: 'sage' and 'spectral'.")
    
    return embedder


# Test PairData
if __name__ == '__main__':


    x_s = torch.randn(5, 16)  # 5 nodes
    edge_index_s = torch.tensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ])

    x_t = torch.randn(4, 16)  # 4 nodes
    edge_index_t = torch.tensor([
        [0, 0, 0],
        [1, 2, 3],
    ])

    edge_weight_s = torch.rand(4)
    edge_weight_t = torch.rand(3)
    edge_attr_s = torch.rand(4, 100)
    edge_attr_t = torch.rand(3, 100)

    num_nodes_s = 5
    num_nodes_t = 4

    data = PairData(x_s=x_s, edge_index_s=edge_index_s, edge_weight_s=edge_weight_s, edge_attr_s=edge_attr_s,# num_nodes_s=num_nodes_s,
                    x_t=x_t, edge_index_t=edge_index_t, edge_weight_t=edge_weight_t, edge_attr_t=edge_attr_t)#, num_nodes_t=num_nodes_t)

    data_list = [data, data, data, data]
    loader = DataLoader(data_list, batch_size=3, follow_batch=['x_s', 'x_t', 'edge_weight_s', 'edge_weight_t', 'edge_attr_s', 'edge_attr_t'])
    batch = next(iter(loader))

    print("batch:", batch)

    print("edge_index_s:", batch.edge_index_s)
    print("x_s_batch:", batch.x_s_batch)
    print("x_s_ptr:", batch.x_s_ptr)
    print("edge_attr_s_batch:", batch.edge_attr_s_batch)
