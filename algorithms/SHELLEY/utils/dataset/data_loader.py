"""
Dataset class and DataLoader utilities adapted from 
ThinkMatch: https://github.com/Thinklab-SJTU/ThinkMatch
"""


import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_networkx

from input.dataset import Dataset as BaseDataset

cfg = edict()   # global variable that substitue a configuration file


class GMSemiSyntheticDataset(Dataset):
    def __init__(self,
                 source_dataset,
                 p_add_connection=0.0,
                 p_remove_connection=0.0,
                 length=1000,
                 mode='train',
                 fix_seed=True,
                 seed=42):
        
        super(GMSemiSyntheticDataset, self).__init__()
        self.source_dataset = source_dataset        # using the base `Dataset` class
        self.p_add_connection = p_add_connection
        self.p_remove_connection = p_remove_connection
        self.length = length
        self.mode = mode
        self.fix_seed = fix_seed
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get_pair(idx)
    
    @staticmethod
    def synthetic_random_clone(dataset, p_add_connection=0.0, p_remove_connection=0.0, seed=42):
        np.random.seed = seed
        
        H = dataset.G.copy()

        adj = dataset.get_adjacency_matrix()
        adj *= np.tri(*adj.shape)

        idx2id = {v: k for k,v in dataset.id2idx.items()}
        connected = np.argwhere(adj==1)
        unconnected = np.argwhere(adj==0)

        # The number of possible new edges is much
        # higher than the number of actual edges.
        # We need to rescale the probability of 
        # adding a new connection accordingly.
        scale_factor = (len(connected) / len(unconnected)) 

        # Remove edges with probability
        mask_remove = np.random.uniform(0,1, size=(len(connected))) < p_remove_connection
        edges_remove = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(connected)
                        if mask_remove[idx] is True]
        # count_rm = mask_remove.sum()
        H.remove_edges_from(edges_remove)

        # Add edges with probability
        mask_add = np.random.uniform(0,1, size=(len(unconnected))) < p_add_connection * scale_factor
        edges_add = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(unconnected)
                    if mask_add[idx] is True]
        # count_add = mask_add.sum()
        H.add_edges_from(edges_add)

        return H
    

    @staticmethod
    def to_pyg_graph(G, node_feats=None, node_metric='degree', edge_feats=None):
        
        # Set node attributes
        if node_feats is None:
            # Generate node features using a local metric
            node_feats = []
            if node_metric == 'degree':
                node_feats = dict(G.degree())
            elif node_metric == 'assortativity':
                node_feats = nx.degree_assortativity_coefficient(G)
            elif node_metric == 'page_rank':
                node_feats = nx.pagerank(G)
            else:
                raise ValueError(f"Invalid node metric {node_metric}.")
        
        nx.set_node_attributes(G, node_feats, name='x')

        # Set edge attributes
        if edge_feats is None:
            if nx.is_weighted(G):
                # If weighted, add weight as an edge attribute
                edge_weights = dict(G.edges(data='weight'))
                nx.set_edge_attributes(G, edge_weights)
            else:
                # Otherwise, use the jaccard coefficient
                jaccard_coefficient = nx.jaccard_coefficient(G)
                edge_jaccard = {(u, v): coef for (u, v, coef) in jaccard_coefficient}
                nx.set_edge_attributes(G, edge_jaccard)
        
        # Convert the so obtained graph to PyG
        pyg_graph = from_networkx(G,
                                  group_node_attrs=all,
                                  group_edge_attrs=all)
            
        return pyg_graph


    def get_pair(self, idx):
        """
        Return a dictionary the all the informations
        of a source-target pair.
        - 'source' is always the graph generated from the 'source_dataset'.
        - 'target' is a random clone with noise of the source graph.
        """
        # Get source graph
        source_graph = self.source_dataset.G

        # Generate random synthetic target graph
        target_graph = self.synthetic_random_clone(self.source_dataset,
                                                   p_add_connection=self.p_add_connection,
                                                   p_remove_connection=self.p_remove_connection,
                                                   seed=self.seed)

        # Convert to pyg graphs
        source_pyg = self.to_pyg_graph(source_graph)
        target_pyg = self.to_pyg_graph(target_graph)
        
        # Get the number of graph nodes
        n_src = source_graph.number_of_nodes()
        n_tgt = target_graph.number_of_nodes()

        # Assemble the informations in a dictionary
        ret_dict = edict()
        ret_dict = {'pyg_graphs': [source_pyg, target_pyg],
                    'ns': [torch.tensor(x) for x in [n_src, n_tgt]]
                    }
        
        return ret_dict

def collate_fn(data):
    """
    Create mini-batch data for training.
    
    Args:
        data:   the data dictionary returned by the
                graph matchin dataset.
    
    Returns:    the mini-batch
    """
   
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            #pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if isinstance(inp[0], list):
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif isinstance(inp[0], dict):
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                ret[k] = stack(vs)
        elif isinstance(inp[0], torch.Tensor):
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif isinstance(inp[0], np.ndarray):
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif isinstance(inp[0], pyg.data.Data):
            ret = pyg.data.Batch.from_data_list(inp)
        elif isinstance(inp[0], str):
            ret = inp
        elif isinstance(inp[0], tuple):
            ret = inp

        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    ret['batch_size'] = len(data)

    for v in ret.values():
        if isinstance(v, list):
            ret['num_graphs'] = len(v)
            break

    return ret

def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False, batch_size=1, num_workers=1, seed=42):
    # Set seed globally
    cfg.RANDOM_SEED = seed

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


if __name__ == '__main__':
    # Get sample dataset
    data_dir = 'dataspace/ppi/graphsage'
    source_dataset = BaseDataset(data_dir)

    # Test semi synt graph matching dataset
    gm_dataset = GMSemiSyntheticDataset(source_dataset=source_dataset,
                                        p_add_connection=0.1,
                                        p_remove_connection=0.1,
                                        length=10,
                                        mode='train')
    
    dataloader = get_dataloader(gm_dataset, batch_size=4)
    
    for idx, ret in enumerate(dataloader):
        print(f'Iter: {idx}')
        print(ret)
        print()