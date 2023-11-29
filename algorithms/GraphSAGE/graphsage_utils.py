from torch_geometric.utils.convert import from_networkx

def networkx_to_pyg(dataset):
    """
    Convert a NetworkX graph in a edge-list representation 
    useful for PyG.
    """
    G = dataset.G
    id2idx = dataset.id2idx
    node_feats = dataset.features
    edge_feats = dataset.edge_features

    print(f"node_feats: {node_feats.shape}")
    print(f"edge_feats: {edge_feats.shape}")

    data = from_networkx(G)

    return data