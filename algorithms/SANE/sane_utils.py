import numpy as np
from torch_geometric.utils.convert import from_networkx
from algorithms.SANE.prediction_models import *
from algorithms.SANE.embedding_models import *

def networkx_to_pyg(dataset):
    """
    Convert a NetworkX graph in a edge-list representation 
    useful for PyG.
    """
    # Get network informations from the dataset
    G = dataset.G
    id2idx = dataset.id2idx
    idx2id = {idx: id for id, idx in id2idx.items()}
    node_feats = dataset.features.astype(np.float32)
    edge_feats = dataset.edge_features
    
    # # Add node features to the network...
    # if node_feats is not None:
    #     for id, idx in id2idx.items():
    #         G.nodes[id]['x'] = node_feats[idx]

    # # ... and edge features
    # # for u, v in G.edges:
    # #     idx_u = idx2id[u]
    # #     idx_v = idx2id[v]
    # #     G.edges[(u, v)]['w'] = edge_feats[:, idx_u, idx_v]
    
    # Generate PyG representation
    data = from_networkx(G)
    data.x = node_feats
    data.edge_attr = edge_feats

    return data, idx2id


def get_prediction_model(model, input_dim=0):
    if model == 'cosine_similarity':
        predictor = CosineSimilarity()
    elif model == 'inner_product':
        predictor = InnerProduct()
    elif model == 'DNN':
        predictor = LinkPredictor(input_dim=input_dim)
    else:
        raise ValueError(f"{model} is an invalid model!")

    return predictor


def get_embedding_model(model, input_dim=None, hidden_dim=None, output_dim=None, num_layers=None):
    return 