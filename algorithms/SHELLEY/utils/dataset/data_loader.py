"""
Dataset class and DataLoader utilities adapted from 
ThinkMatch: https://github.com/Thinklab-SJTU/ThinkMatch
"""


import os
import random
from itertools import chain
from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from easydict import EasyDict as edict
from torch.utils.data import Dataset, random_split
from torch_geometric.utils import (add_random_edge, contains_self_loops,
                                   dropout_edge, is_undirected,
                                   remove_self_loops, shuffle_node,
                                   to_undirected)
from torch_geometric.utils.convert import from_networkx

from input.dataset import Dataset as BaseDataset
from utils.graph_utils import load_gt

cfg = edict()   # Global variable as configuration file


def compute_node_metric(G, metric: str):
    if metric == 'degree':
        return dict(G.degree())
    if metric == 'pagerank':
        return nx.pagerank(G)
    raise ValueError(f"Invalid node metric name {metric}.")
    
def compute_edge_metric(G, metric):
    if metric == 'betwenneess':
        return nx.edge_betweenness_centrality(G, weight='weight')
    raise ValueError(f"Invalid edge metric name {metric}.")


class GMSemiSyntheticDataset(Dataset):
    def __init__(self, root_dir, p_add=0.0, p_rm=0.0,
                 self_supervised=False, size=100, seed=42):
        
        super(GMSemiSyntheticDataset, self).__init__()

        self.source_dataset = BaseDataset(root_dir + '/graphsage', verbose=False)   
        self.p_add = p_add                      # if -1, use all probs
        self.p_rm = p_rm                        # if -1, use all probs
        self.self_supervised = self_supervised  # generate a random target graph 'on-the-fly'
        self.seed = seed 
        
        if not self_supervised:
            # Get from the `root_dir` only the target graph with the
            # specified percentage of noise
            self.targets_dir = os.path.join(root_dir, 'targets')
            p_add_str = f'add{p_add:.2f}'.replace('0.', '') if p_add >= 0 else ''
            p_rm_str = f'rm{p_rm:.2f}'.replace('0.', '') if p_rm >= 0 else ''
            prob_substrings = [p_add_str, p_rm_str]
            self.target_files = sorted(self.filter_files_with_substrings(self.targets_dir, prob_substrings))
        else:
            # Set the number of random generated graphs by hand
            self.size = size
            
    def __len__(self):
        if self.self_supervised:
            return self.size
        else:
            return len(self.target_files)

    def __getitem__(self, idx):
        return self.get_pair(idx)
    
    @staticmethod
    def filter_files_with_substrings(directory, substring_list):
        matching_files = []

        # List all files in the directory
        files = os.listdir(directory)

        for file in files:
            # Check if the file name contains all substrings in the substring list
            if all(substring in file for substring in substring_list):
                matching_files.append(file)

        return matching_files
        
    @staticmethod
    def generate_synth_clone(source_pyg, p_rm=0.0, p_add=0.0):
        # Clone source pyg graph
        target_pyg = source_pyg.clone() 
        
        # Check if undirected
        if is_undirected(source_pyg.edge_index):
            force_undirected = True      
        
        # Remove edges with probability
        target_pyg.edge_index, edge_mask = dropout_edge(target_pyg.edge_index,
                                                        p=p_rm,
                                                        force_undirected=force_undirected)
        target_pyg.edge_attr = target_pyg.edge_attr[edge_mask]
        
        # Add edges with probability
        target_pyg.edge_index, added_edges = add_random_edge(target_pyg.edge_index,
                                                             p=p_add,
                                                             force_undirected=force_undirected)
        
        # Sample edge attributes to assign them to the new added edge index.
        # Sampling consent us to obtain not trivial attributes values
        # easily recognisable from a GCN.
        num_new_edges = added_edges.size(1)
        sample_indices = torch.randperm(target_pyg.edge_attr.size(0))[:num_new_edges]
        old_attr = target_pyg.edge_attr
        new_attr = target_pyg.edge_attr[sample_indices]
        target_pyg.edge_attr = torch.cat((old_attr, new_attr), dim=0)

        # Shuffle nodes
        target_pyg.x, node_perm = shuffle_node(target_pyg.x)
        
        # Make it undirected and remove self loops
        if force_undirected is True and not is_undirected(target_pyg.edge_index):
            target_pyg.edge_index, target_pyg.edge_attr = to_undirected(target_pyg.edge_index, target_pyg.edge_attr)
        if contains_self_loops(target_pyg.edge_index):
            target_pyg.edge_index, target_pyg.edge_attr = remove_self_loops(target_pyg.edge_index, target_pyg.edge_attr)
            
        # Build the groundtruth matrix
        gt_perm_mat = torch.zeros((source_pyg.num_nodes, target_pyg.num_nodes), dtype=torch.float)
        for s, t in enumerate(node_perm):   # `node_perm` contains the order of original nodes after shuffling
            gt_perm_mat[s, t] = 1
        
        # Build the groundtruth permuted index tensor
        source_indices = torch.arange(0, source_pyg.num_nodes)
        target_indices = node_perm
        gt_perm_edge_index = torch.stack((source_indices, target_indices))
        
        return target_pyg, gt_perm_mat, gt_perm_edge_index
    

    @staticmethod
    def to_pyg_graph(G,
                     id2idx: dict,
                     node_feats: torch.Tensor=None,
                     edge_feats: torch.Tensor=None,
                     node_metrics: List[str]=[],
                     edge_metrics: List[str]=[]):
        
        # Assign existing features
        if node_feats is not None:
            for n in G.nodes:
                G.nodes[n]['features'] = torch.Tensor(node_feats[id2idx[n]])
        
        if edge_feats is not None:
            for e, f in zip(G.edges, edge_feats):
                G.edges[e]['features'] = torch.Tensor(f)
                
        # Explicit edge weight to 1 if unweighted
        if not nx.is_weighted(G):
            nx.set_edge_attributes(G, 1, name='weight')
                
        # Generate new node/edge features using local metrics
        if len(node_metrics) > 0:
            for metric in node_metrics:
                feats = compute_node_metric(G, metric)
                nx.set_node_attributes(G, feats, name=metric) 
                
        if len(edge_metrics) > 0:
            for metric in edge_metrics:
                feats = compute_edge_metric(G, metric)
                nx.set_edge_attributes(G, feats, name=metric)
        
        
        # Get the list of node/edge attribute names
        node_attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))
        edge_attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
        
        # Convert to pyg
        pyg_graph = from_networkx(G,
                                  group_node_attrs=node_attrs_list,
                                  group_edge_attrs=edge_attrs_list)

        # Set default node importance to 1 if not present yet
        if 'x_importance' not in pyg_graph:
            pyg_graph.x_importance = torch.ones([pyg_graph.num_nodes, 1])
            
        # Make it undirected and remove self loops
        if not nx.is_directed(G) and not is_undirected(pyg_graph.edge_index):
            pyg_graph.edge_index, pyg_graph.edge_attr = to_undirected(pyg_graph.edge_index, pyg_graph.edge_attr)
        if contains_self_loops(pyg_graph.edge_index):
            pyg_graph.edge_index, pyg_graph.edge_attr = remove_self_loops(pyg_graph.edge_index, pyg_graph.edge_attr)
                   
        
        return pyg_graph


    def get_pair(self, idx):
        """
        Return a dictionary with a source, target graph pair and all
        the relative informations.
        'source' is always the graph generated from the 'source_dataset'.
        'target' is a random clone with noise of the source graph.
        
        If `self_supervised` is True, the target graph is randomly generated
        on-the-fly, otherwise it is loaded from the`root_dir`.
        """
        
        if self.self_supervised:
            # Source pyg graph            
            source_pyg = self.to_pyg_graph(G=self.source_dataset.G,
                                           id2idx=self.source_dataset.id2idx,
                                           node_feats=self.source_dataset.features,
                                           edge_feats=self.source_dataset.edge_features,
                                           node_metrics=['degree'],
                                           edge_metrics=[])
            
            # Random semi-synthetic pyg target graph
            target_pyg, gt_perm_mat, gt_perm_edge_index = self.generate_synth_clone(source_pyg,
                                                                                    p_add=self.p_add,
                                                                                    p_rm=self.p_rm)
            
            
        else:           
            # Get graph pair
            source_graph = self.source_dataset.G
            target_path = os.path.join(self.targets_dir, self.target_files[idx], 'graphsage')
            target_dataset = BaseDataset(target_path, verbose=False)
            target_graph = target_dataset.G

            # Get groundtruth alignment matrix
            gt_path = os.path.join(self.targets_dir, self.target_files[idx], 'dictionaries', 'groundtruth')
            gt_perm_mat = torch.from_numpy(load_gt(gt_path,
                                                   self.source_dataset.id2idx,
                                                   target_dataset.id2idx,
                                                   'matrix')).to(torch.float)

            # Convert to pyg graphs
            source_pyg = self.to_pyg_graph(source_graph, node_metric='degree', edge_metric='none', use_weight=True)
            target_pyg = self.to_pyg_graph(target_graph, node_metric='degree', edge_metric='none', use_weight=True)

        # Assemble pair informations in a dictionary
        ret_dict = edict()
        n_src = source_pyg.num_nodes
        n_tgt = target_pyg.num_nodes
        ret_dict = {'pyg_graphs': [source_pyg, target_pyg],
                    'ns': [torch.tensor(x) for x in [n_src, n_tgt]],
                    'gt_perm_mat': gt_perm_mat}
        
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
    # Set seed globally
    cfg.RANDOM_SEED = seed

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


if __name__ == '__main__':
    # Test semi synt graph matching dataset
    data_dir = 'dataspace/edi3'
    gm_dataset = GMSemiSyntheticDataset(root_dir=data_dir,
                                        p_add=0.2,
                                        p_rm=0.2,
                                        self_supervised=True,
                                        size=100)
    
    print("Full dataset:", len(gm_dataset))

    # Split it
    train, val, test = random_split(gm_dataset, (0.7, 0.15, 0.15))
    print('Train:', len(train))
    print('Val:', len(val))
    print('Test', len(test))

    # Test dataloader
    dataloader = get_dataloader(val, batch_size=1)
    for idx, ret in enumerate(dataloader):
        print(f'Iter: {idx}')
        print(ret)
        print()
        exit()