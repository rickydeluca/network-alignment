import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def generate_node_attrs(G, metric="degree", rand_dim=1):
    """
    Given a networkx graph `G`, return the tensor with
    the `metric` values for the node in G.
    """

    if metric == "degree":
        degrees = [deg for (node, deg) in G.degree()]
        return torch.tensor(degrees).unsqueeze(1)
    elif metric == "pagerank":
        pagerank = nx.pagerank(G)
        pagerank_list = [rank for rank in pagerank.values()]
        return torch.tensor(pagerank_list).unsqueeze(1)
    else:
        raise Exception(f"Invalid node metric: {metric}.")


def generate_edge_attrs(G, metric="rand", rand_dim=1):
    if metric == "rand":
        return torch.rand((G.number_of_edges(), rand_dim), dtype=torch.float)
    else:
        raise Exception(f"Invalid edge metric: {metric}.")


def networkx_to_pyg(
    G,
    node_feats=None,
    edge_feats=None,
    normalize=False,
    gen_node_feats=False,
    ref_node_metric="degree",
    gen_edge_feats=False,
    ref_edge_metric="rand",
):
    pyg_graph = from_networkx(G)

    # node features
    if node_feats is not None:
        x = node_feats
        if normalize:
            x = normalize_over_channels(x)
    else:
        if gen_node_feats:
            # generate the node features using the specified metric
            x = generate_node_attrs(G, metric=ref_node_metric)
        else:
            x = None

    # edge features
    if edge_feats is not None:
        edge_attr = edge_feats
        if normalize:
            edge_attr = normalize_over_channels(edge_attr)
    else:
        if gen_edge_feats:
            # generate the edge features using the specified metric
            edge_attr = generate_edge_attrs(G, metric=ref_edge_metric, rand_dim=1)
        else:
            edge_attr = None

    pyg_graph.x = x
    pyg_graph.edge_attr = edge_attr

    return pyg_graph
