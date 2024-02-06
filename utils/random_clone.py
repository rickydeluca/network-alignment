from __future__ import division, print_function

import numpy as np


def random_clone_synthetic(dataset, p_new_connection, p_remove_connection, seed):
    np.random.seed = seed
    H = dataset.G.copy()
    adj = dataset.get_adjacency_matrix()
    adj *= np.tri(*adj.shape)

    idx2id = {v: k for k,v in dataset.id2idx.items()}
    connected = np.argwhere(adj==1)

    mask_remove = np.random.uniform(0,1, size=(len(connected))) < p_remove_connection
    edges_remove = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(connected)
                    if mask_remove[idx] == True]
    # count_rm = mask_remove.sum()
    H.remove_edges_from(edges_remove)

    print("New graph:")
    print("- Number of nodes:", len(H.nodes()))
    print("- Number of edges:", len(H.edges()))
    return H


def random_clone_synthetic_shelley(dataset,
                                   p_add_connection=None,
                                   p_remove_connection=None,
                                   seed=None,
                                   weighted=False):
    np.random.seed = seed
    
    H = dataset.G.copy()

    adj = dataset.get_adjacency_matrix()
    adj *= np.tri(*adj.shape)

    idx2id = {v: k for k,v in dataset.id2idx.items()}
    connected = np.argwhere(adj==1)
    unconnected = np.argwhere(adj==0)

    # Remove edges with probability.
    mask_remove = np.random.uniform(0,1, size=(len(connected))) < p_remove_connection
    edges_remove = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(connected)
                    if mask_remove[idx] == True]

    H.remove_edges_from(edges_remove)

    # The number of possible new edges is much higher
    # than the number of actual edges.
    # We need first to compute the number of expected new
    # edges according the total number of existing connections.
    num_existing_edges = connected.shape[0]
    num_new_edges = int(num_existing_edges * p_add_connection)
    edges_add = []
    for _ in range(num_new_edges):
        already_connected = True
        connected_new_edges = []

        while already_connected:
            random_index = np.random.choice(unconnected.shape[0])
            random_edge = unconnected[random_index]

            if random_edge not in connected_new_edges:
                already_connected = False
                connected_new_edges.append(random_edge)
        
        if weighted:
            # Generate a random weight between -1 and +1.
            u, v = random_edge
            w = np.random.uniform(-1, 1)
            edges_add.append((idx2id[u], idx2id[v], w))
        else:
            u, v = random_edge
            edges_add.append((idx2id[u], idx2id[v]))

    # Add the new edges to the graph.
    if weighted:
        H.add_weighted_edges_from(edges_add)
    else:
        H.add_edges_from(edges_add)

    print("")
    print("Noising the graph:")
    print("Number of added edges: ", len(edges_add))
    print("Number of removed edges: ", len(edges_remove))
    print("")

    print("")
    print("New Graph:")
    print(H)
    print("")

    return H