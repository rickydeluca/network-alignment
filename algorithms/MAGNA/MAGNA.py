import argparse
import os

import numpy as np

from algorithms.MAGNA.edgelist_utils import read_align_dict
from algorithms.network_alignment_model import NetworkAlignmentModel
from input.dataset import Dataset


class MAGNA(NetworkAlignmentModel):
    def __init__(self, source_dataset=None, target_dataset=None, source_edgelist=None, target_edgelist=None, measure="S3", population_size=15000, num_generations=2000, num_threads=8, outfile="algorithms/MAGNA/output/magna", reverse=False):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_edgelist = source_edgelist
        self.target_edgelist = target_edgelist
        self.measure = measure
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_threads = num_threads
        self.outfile = outfile
        self.reverse = reverse

    def align(self):
        
        # Usually the inputs of MAGNA should in reverse order 
        # wrt the standard convention.
        # If the `reverse` flag is True, switch the edgelists.
        # At the end it is important to read also the alignment produced 
        # by MAGNA in reverse order.
        if self.reverse:
            G_edgelist = self.target_edgelist
            H_edgelist = self.source_edgelist
        else:
            G_edgelist = self.source_edgelist
            H_edgelist = self.target_edgelist

        # Run MAGNA++
        os.system(f"algorithms/MAGNA/magnapp_cli_linux64 -G {G_edgelist} -H {H_edgelist} -m {self.measure} -p {self.population_size} -n {self.num_generations} -t {self.num_threads} -o {self.outfile}")
        
        # Get networks
        G = self.source_dataset.G
        H = self.target_dataset.G
        G_id2idx = self.source_dataset.id2idx
        H_id2idx = self.target_dataset.id2idx

        # Read output alignment
        align_dict, weighted = read_align_dict(f"{self.outfile}_final_alignment.txt", reverse=self.reverse)

        # Generate alignment matrix
        S = np.zeros((G.number_of_nodes(), H.number_of_nodes()))

        if weighted:
            for source, target_and_weight in align_dict.items():
                target, weight = target_and_weight
                source_idx = G_id2idx[source]
                target_idx = H_id2idx[target]
                S[source_idx, target_idx] = weight
        
        else:
            for source, target in align_dict.items():
                source_idx = G_id2idx[source]
                target_idx = H_id2idx[target]
                S[source_idx, target_idx] = 1

        self.alignment_matrix = S

        return self.alignment_matrix


    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix



def parse_args():
    parser = argparse.ArgumentParser(description="HDA")
    parser.add_argument('--prefix1', default="dataspace/douban/online/graphsage")
    parser.add_argument('--prefix2', default="dataspace/douban/offline/graphsage")
    parser.add_argument('--source_edgelist', default="dataspace/douban/offline/edgelist/edgelist", type=str)
    parser.add_argument('--target_edgelist', default="dataspace/douban/online/edgelist/edgelist", type=str)
    parser.add_argument('--measure', default="S3", type=str)
    parser.add_argument('--population_size', default=200, type=int)
    parser.add_argument('--num_generations', default=100, type=int)
    parser.add_argument('--num_threads', default=8, type=int)
    parser.add_argument('--outfile', default="algorithms/MAGNA/output/magna", type=str)
    parser.add_argument('--reverse', action="store_true", default=False)

    return parser.parse_args()


def main(args):

    model = MAGNA(
        source_dataset=Dataset(args.prefix1),
        target_dataset=Dataset(args.prefix2),
        source_edgelist=args.source_edgelist,
        target_edgelist=args.target_edgelist,
        measure=args.measure,
        population_size=args.population_size,
        num_generations=args.num_generations,
        num_threads=args.num_threads,
        outfile=args.outfile,
        reverse=args.reverse
    )
    
    S = model.align()

    print(S)

if __name__ == "__main__":
    args = parse_args()
    main(args)