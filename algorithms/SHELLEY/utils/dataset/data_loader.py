"""
Dataset class and DataLoader utilities adapted from 
ThinkMatch: https://github.com/Thinklab-SJTU/ThinkMatch
"""


import glob
import os
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
from utils.graph_utils import load_gt

cfg = edict()   # Global variable that substitue a configuration file


class GMSemiSyntheticDataset(Dataset):
    def __init__(self,
                 root_dir,
                 p_add_connection=0.0,
                 p_remove_connection=0.0,
                 mode='train',
                 fix_seed=True,
                 seed=42):
        
        super(GMSemiSyntheticDataset, self).__init__()

        self.source_dataset = BaseDataset(root_dir + '/graphsage', verbose=False)   
        self.p_add_connection = p_add_connection
        self.p_remove_connection = p_remove_connection
        self.mode = mode
        self.fix_seed = fix_seed
        self.seed = seed

        # Take only the target graphs with the required probabilities.
        self.targets_dir = os.path.join(root_dir, 'targets')
        p_add_str = f'add{p_add_connection:.2f}'.replace('0.', '')
        p_rm_str = f'rm{p_add_connection:.2f}'.replace('0.', '')
        prob_substrings = [p_add_str, p_rm_str]
        self.target_files = sorted(self.filter_files_with_substrings(self.targets_dir, prob_substrings))

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, idx):
        return self.get_pair(idx)
    
    @staticmethod
    def filter_files_with_substrings(directory, substring_list):
        matching_files = []

        # List all files in the directory.
        files = os.listdir(directory)

        for file in files:
            # Check if the file name contains all substrings in the substring list.
            if all(substring in file for substring in substring_list):
                matching_files.append(file)

        return matching_files
        
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

        # Remove edges with probability.
        mask_remove = np.random.uniform(0,1, size=(len(connected))) < p_remove_connection
        edges_remove = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(connected)
                        if mask_remove[idx] is True]
        # count_rm = mask_remove.sum()
        H.remove_edges_from(edges_remove)

        # Add edges with probability.
        mask_add = np.random.uniform(0,1, size=(len(unconnected))) < p_add_connection * scale_factor
        edges_add = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(unconnected)
                    if mask_add[idx] is True]
        # count_add = mask_add.sum()
        H.add_edges_from(edges_add)

        return H
    

    @staticmethod
    def to_pyg_graph(G, node_feats=None, node_metric='degree', edge_feats=None, edge_metric='none', use_weight=True):
        
        # Generate node attributes.
        if node_feats is None:           
            if node_metric == 'all':
                node_metrics = ['degree', 'assortativity', 'pagerank']
            elif node_metric == 'none':
                node_metrics = []
            else:
                node_metrics = [node_metric]

            if 'degree' in node_metrics:
                node_feats = dict(G.degree())
                nx.set_node_attributes(G, node_feats, name='degree')
            if 'assortativity' in node_metrics:
                node_feats = nx.degree_assortativity_coefficient(G)
                nx.set_node_attributes(G, node_feats, name='assortativity')
            if 'pagerank' in node_metrics:
                node_feats = nx.pagerank(G)
                nx.set_node_attributes(G, node_feats, name='pagerank')

        # Generate edge attributes.
        if edge_feats is None:
            edge_feats = []

            if edge_metric == 'all':
                edge_metrics = ['betweenness']
            elif edge_metric == 'none':
                edge_metrics = ['none']
            else:
                edge_metrics = [edge_metric]

            # If weighted graph, use also weight as attribute.
            if nx.is_weighted(G) and use_weight is True:
                # The weight attribute is already in the graph
                # and we don't have to recompute it.
                edge_metrics.append('weight')
    
            if 'betweenness' in edge_metrics:
                edge_feats = nx.edge_betweenness_centrality(G, normalized=False)
                nx.set_edge_attributes(G, edge_feats, name='betweenness')

            if 'none' in edge_metrics:
                nx.set_edge_attributes(G, 1, name='none')

        # Convert the so obtained graph to pyg.
        pyg_graph = from_networkx(G,
                                  group_node_attrs=node_metrics,
                                  group_edge_attrs=edge_metrics)
            
        return pyg_graph


    def get_pair(self, idx):
        """
        Return a dictionary the all the informations
        of a source-target pair.
        - 'source' is always the graph generated from the 'source_dataset'.
        - 'target' is a random clone with noise of the source graph.
        """
        # Get graph pair.
        source_graph = self.source_dataset.G
        target_path = os.path.join(self.targets_dir, self.target_files[idx], 'graphsage')
        target_dataset = BaseDataset(target_path, verbose=False)
        target_graph = target_dataset.G

        # Get groundtruth alignment matrix.
        gt_path = os.path.join(self.targets_dir, self.target_files[idx], 'dictionaries', 'groundtruth')
        groundtruth = torch.from_numpy(load_gt(gt_path, self.source_dataset.id2idx, target_dataset.id2idx, 'matrix'))

        # Generate random synthetic target graph.
        # target_graph = self.synthetic_random_clone(self.source_dataset,
        #                                            p_add_connection=self.p_add_connection,
        #                                            p_remove_connection=self.p_remove_connection,
        #                                            seed=self.seed)

        # Convert to pyg graphs.
        source_pyg = self.to_pyg_graph(source_graph, node_metric='degree', edge_metric='none', use_weight=True)
        target_pyg = self.to_pyg_graph(target_graph, node_metric='degree', edge_metric='none', use_weight=True)

        # Get the number of graph nodes.
        n_src = source_graph.number_of_nodes()
        n_tgt = target_graph.number_of_nodes()

        # Assemble the informations in a dictionary.
        ret_dict = edict()
        ret_dict = {'pyg_graphs': [source_pyg, target_pyg],
                    'ns': [torch.tensor(x) for x in [n_src, n_tgt]],
                    'gt_perm_mat': groundtruth}
        
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
            # pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
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
    # Set seed globally.
    cfg.RANDOM_SEED = seed

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


if __name__ == '__main__':
    # Test semi synt graph matching dataset.
    data_dir = 'dataspace/ppi'
    gm_dataset = GMSemiSyntheticDataset(root_dir=data_dir,
                                        p_add_connection=0,
                                        p_remove_connection=0,
                                        mode='train')
    
    dataloader = get_dataloader(gm_dataset, batch_size=4)
    
    for idx, ret in enumerate(dataloader):
        print(f'Iter: {idx}')
        print(ret)
        print()