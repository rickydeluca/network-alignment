import argparse

import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE


from algorithms.network_alignment_model import NetworkAlignmentModel
from input.dataset import Dataset
from algorithms.GraphSAGE.graphsage_utils import networkx_to_pyg

class GraphSAGE(NetworkAlignmentModel):
    def __init__(self, source_dataset=None, target_dataset=None, embedding_size=None):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.embedding_size = embedding_size
        
    def align(self):
        # Convert the NetworkX networks to the edge list format for PyG
        data1 = networkx_to_pyg(self.source_dataset)
        data2 = networkx_to_pyg(self.target_dataset)
        print(f"data1: {data1}")
        print(f"data2: {data2}")
        
        return 0
    
    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument('--prefix1', default="dataspace/douban/online/graphsage")
    parser.add_argument('--prefix2', default="dataspace/douban/offline/graphsage")
    parser.add_argument('--embedding_size', type=int, default=1000)
    parser.add_argument('--groundtruth', default=None)

    return parser.parse_args()


def main(args):

    model = GraphSAGE(
        source_dataset=Dataset(args.prefix1),
        target_dataset=Dataset(args.prefix2),
        embedding_size=args.embedding_size,
    )
    
    S = model.align()

    print(S)

if __name__ == "__main__":
    args = parse_args()
    main(args)
