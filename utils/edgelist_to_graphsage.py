from collections import defaultdict
import argparse
import numpy as np
import json
import os
import networkx as nx
from networkx.readwrite import json_graph

def parse_args():
    parser = argparse.ArgumentParser(description="Generate graphsage format from edgelist")
    parser.add_argument('--out_dir', default=None, help="Output directory")
    parser.add_argument('--weighted', action="store_true", help="Weighted edgelist")
    parser.add_argument('--seed', default=121, type=int)
    return parser.parse_args()

def network_info(G, n=None):
    """Print short summary of information for the graph G or the node n.

    Parameters
    ----------
    G : Networkx graph
       A graph
    n : node (any hashable)
       A node in the graph G
    """
    info='' # append this all to a string
    if n is None:
        info+="Name: %s\n"%G.name
        type_name = [type(G).__name__]
        info+="Type: %s\n"%",".join(type_name)
        info+="Number of nodes: %d\n"%G.number_of_nodes()
        info+="Number of edges: %d\n"%G.number_of_edges()
        nnodes=G.number_of_nodes()
        if len(G) > 0:
            if G.is_directed():
                info+="Average in degree: %8.4f\n"%\
                    (sum(G.in_degree().values())/float(nnodes))
                info+="Average out degree: %8.4f"%\
                    (sum(G.out_degree().values())/float(nnodes))
            else:
                degrees = dict(G.degree())
                s=sum(degrees.values())
                info+="Average degree: %8.4f"%\
                    (float(s)/float(nnodes))

    else:
        if n not in G:
            raise nx.NetworkXError("node %s not in graph"%(n,))
        info+="Node % s has the following properties:\n"%n
        info+="Degree: %d\n"%G.degree(n)
        info+="Neighbors: "
        info+=' '.join(str(nbr) for nbr in G.neighbors(n))

    return info


def edgelist_to_graphsage(dir, weighted=False, seed=121):
    np.random.seed(seed)
    edgelist_dir = dir + "/edgelist/edgelist"
    print(edgelist_dir)

    if weighted:
        G = nx.read_weighted_edgelist(edgelist_dir)
    else:
        G = nx.read_edgelist(edgelist_dir)
    
    print(network_info(G))
    num_nodes = len(G.nodes())
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.81)]
    val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
    test = rand_indices[int(num_nodes * 0.9):]

    id2idx = {}
    for i, node in enumerate(G.nodes()):
        id2idx[str(node)] = i

    res = json_graph.node_link_data(G)
    res['nodes'] = [
        {
            'id': node['id'],
            'val': id2idx[str(node['id'])] in val,
            'test': id2idx[str(node['id'])] in test
        }
        for node in res['nodes']]

    if weighted:
        res['links'] = [
            {
                'source': link['source'],
                'target': link['target'],
                'weight': link['weight']
            }
            for link in res['links']]
    else:
        res['links'] = [
            {
                'source': link['source'],
                'target': link['target']
            }
            for link in res['links']] 

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")

    with open(dir + "/graphsage/" + "G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + "id2idx.json", 'w') as outfile:
        json.dump(id2idx, outfile)

    print("GraphSAGE format stored in {0}".format(dir + "/graphsage/"))
    print("----------------------------------------------------------")

if __name__ == "__main__":
    args = parse_args()
    datadir = args.out_dir
    weighted = args.weighted

    edgelist_to_graphsage(datadir, weighted=weighted)



