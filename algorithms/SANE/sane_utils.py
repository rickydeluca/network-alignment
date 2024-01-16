import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from algorithms.HDA.hyperdimensional_encoding import *
from algorithms.SANE.embedding_model import *
from algorithms.SANE.mapping_model import *
from algorithms.SANE.prediction_model import *


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
    """
    Given a `Dataset` object, return the network using the
    pytorch geometric `Data` object.
    """
    # Convert NetworX to Data representation
    G = from_networkx(dataset.G)

    # Extract useful informations
    edge_index = G.edge_index
    edge_weight = G.weight if 'weight' in G.keys() else None

    # Read node/edge features    
    x = torch.tensor(dataset.features, dtype=torch.float32) if dataset.features is not None else None
    edge_attr = torch.tensor(dataset.edge_features) if dataset.edge_features is not None else None

    # Use positional informations if required
    if pos_info:
        adj_matrix = torch.tensor(nx.adjacency_matrix(dataset.G).todense(), dtype=torch.float32)
        if x is not None:
            x = torch.cat((x, adj_matrix), dim=1)

    # Build the Data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

    return data


def get_embedding_model(model, in_channels=None, hidden_channels=None, out_channels=None, num_layers=None, dropout=0.0):
    """
    Return the chosen model to generate node embeddings 
    for unsupervised learning.
    """
    if model == 'sage':
        embedder = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout)
    elif model == 'spectral':
        embedder = SaneEmbeddingGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_hidden_layers=num_layers-1)
    else:
        raise ValueError(f"{model} is an invalid model name. Choose from: 'sage' and 'spectral'.")
    return embedder


def get_prediction_model(model, input_dim=0, dropout=0.5):
    """
    Returns the chosen model to learn link prediction
    for unsupervised learning.
    """
    if model == 'cosine_similarity':
        predictor = CosineSimilarity()
    elif model == 'inner_product':
        predictor = InnerProduct()
    elif model == 'dnn':
        predictor = LinkPredictor(input_dim=input_dim, dropout=dropout)
    else:
        raise ValueError(f"{model} is an invalid model name. Choose from: 'cosine_similarity', 'inner_product', 'dnn'")
    return predictor


def get_mapping_model(model, embedding_dim=None, source_embedding=None, target_embedding=None, heads=None, activate_function='sigmoid'):
    """
    Returns the chosen model to learn mapping between node embeddings
    for supervised learning.
    """
    if model == "linear":
        mapper = PaleMappingLinear(embedding_dim=embedding_dim,
                                   source_embedding=source_embedding,
                                   target_embedding=target_embedding)
        
    elif model == "cross_transformer":
        mapper = PaleMappingCrossTransformer(source_embedding=source_embedding,
                                             target_embedding=target_embedding,
                                             k=embedding_dim,
                                             heads=heads,
                                             activate_function=activate_function)
    elif model == "mlp":
        mapper = PaleMappingMlp(embedding_dim=embedding_dim,
                                source_embedding=source_embedding,
                                target_embedding=target_embedding,
                                activate_function=activate_function)
    else:
        raise ValueError(f"{model} is an invalid mapping molde. Choose from: 'lineae', 'mlp' and 'cross_transformer'.")
    
    return mapper