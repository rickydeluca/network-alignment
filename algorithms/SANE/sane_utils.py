import numpy as np
from torch_geometric.utils.convert import from_networkx
from algorithms.SANE.mapping_model import *
from algorithms.SANE.embedding_model import *
from algorithms.HDA.hyperdimensional_encoding import *

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

    if dataset.edge_features:
        edge_feats = torch.tensor(dataset.edge_features)

    # Generate PyG representation
    data = from_networkx(G)
    data.x = node_feats

    if dataset.edge_features:
        data.edge_attr = edge_feats

    return data #, idx2id


def get_mapping_model(model, input_dim=0, dropout=0.5):
    if model == 'cosine_similarity':
        predictor = CosineSimilarity()
    elif model == 'inner_product':
        predictor = InnerProduct()
    elif model == 'dnn':
        predictor = LinkPredictor(input_dim=input_dim)
    else:
        raise ValueError(f"{model} is an invalid model name. Choose from: 'cosine_similarity', 'inner_product', 'dnn'")

    return predictor


def get_embedding_model(model, in_channels=None, hidden_channels=None, out_channels=None, num_layers=None, dropout=0.0):

    if model == 'sage':
        embedder = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout)
    elif model == 'spectral':
        embedder = SpectralGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_hidden_layers=num_layers-1)
    else:
        raise ValueError(f"{model} is an invalid model name. Choose from: 'sage' and 'spectral'.")
    
    return embedder