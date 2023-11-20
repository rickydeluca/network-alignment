from input.dataset import Dataset
from time import time
from algorithms import *
from evaluation.metrics import get_statistics
import utils.graph_utils as graph_utils
import random
import numpy as np
import pandas as pd
import torch
import argparse
import os
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="dataspace/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="dataspace/douban/offline/graphsage/")
    parser.add_argument('--groundtruth',    default="dataspace/douban/dictionaries/groundtruth")
    parser.add_argument('--alignment_matrix_name', default=None, help="Prefered name of alignment matrix.")
    parser.add_argument('--seed',           default=123,    type=int)

    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: IsoRank, FINAL, UniAlign, PALE, DeepLink, REGAL, IONE, HDA')

    parser_IsoRank = subparsers.add_parser('IsoRank', help='IsoRank algorithm')
    parser_IsoRank.add_argument('--H',                   default=None, help="Priority matrix")
    parser_IsoRank.add_argument('--max_iter',            default=30, type=int, help="Max iteration")
    parser_IsoRank.add_argument('--alpha',               default=0.82, type=float)
    parser_IsoRank.add_argument('--tol',                 default=1e-4, type=float)

    parser_FINAL = subparsers.add_parser('FINAL', help='FINAL algorithm')
    parser_FINAL.add_argument('--H',                   default=None, help="Priority matrix")
    parser_FINAL.add_argument('--max_iter',            default=30, type=int, help="Max iteration")
    parser_FINAL.add_argument('--alpha',               default=0.82, type=float)
    parser_FINAL.add_argument('--tol',                 default=1e-4, type=float)

    parser_BigAlign = subparsers.add_parser('BigAlign', help='BigAlign algorithm')
    parser_BigAlign.add_argument('--lamb', default=0.01, help="Lambda")

    parser_IONE = subparsers.add_parser('IONE', help='IONE algorithm')
    parser_IONE.add_argument('--train_dict', default="groundtruth.train", help="Groundtruth use to train.")
    parser_IONE.add_argument('--epochs', default=100, help="Total iterations.", type=int)
    parser_IONE.add_argument('--dim', default=100, help="Embedding dimension.")


    parser_REGAL = subparsers.add_parser('REGAL', help='REGAL algorithm')
    parser_REGAL.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')

    parser_REGAL.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser_REGAL.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')

    parser_REGAL.add_argument('--max_layer', type=int, default=2,
                        help='Calculation until the layer for xNetMF.')
    parser_REGAL.add_argument('--alpha', type=float, default=0.01, help="Discount factor for further layers")
    parser_REGAL.add_argument('--gammastruc', type=float, default=1, help="Weight on structural similarity")
    parser_REGAL.add_argument('--gammaattr', type=float, default=1, help="Weight on attribute similarity")
    parser_REGAL.add_argument('--num_top', type=int, default=10,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser_REGAL.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")

    parser_PALE = subparsers.add_parser('PALE', help="PALE algorithm")
    parser_PALE.add_argument('--cuda',                action='store_true')

    parser_PALE.add_argument('--learning_rate1',      default=0.01,        type=float)
    parser_PALE.add_argument('--embedding_dim',       default=300,         type=int)
    parser_PALE.add_argument('--batch_size_embedding',default=512,         type=int)
    parser_PALE.add_argument('--embedding_epochs',    default=1000,        type=int)
    parser_PALE.add_argument('--neg_sample_size',     default=10,          type=int)

    parser_PALE.add_argument('--learning_rate2',      default=0.01,       type=float)
    parser_PALE.add_argument('--batch_size_mapping',  default=32,         type=int)
    parser_PALE.add_argument('--mapping_epochs',      default=100,         type=int)
    parser_PALE.add_argument('--mapping_model',       default='linear')
    parser_PALE.add_argument('--activate_function',   default='sigmoid')
    parser_PALE.add_argument('--train_dict',          default='dataspace/douban/dictionaries/node,split=0.2.train.dict')
    parser_PALE.add_argument('--embedding_name',          default='')


    parser_DeepLink = subparsers.add_parser("DeepLink", help="DeepLink algorithm")
    parser_DeepLink.add_argument('--cuda',                action="store_true")

    parser_DeepLink.add_argument('--embedding_dim',       default=800,         type=int)
    parser_DeepLink.add_argument('--embedding_epochs',    default=5,        type=int)

    parser_DeepLink.add_argument('--unsupervised_lr',     default=0.001, type=float)
    parser_DeepLink.add_argument('--supervised_lr',       default=0.001, type=float)
    parser_DeepLink.add_argument('--batch_size_mapping',  default=32,         type=int)
    parser_DeepLink.add_argument('--unsupervised_epochs', default=500, type=int)
    parser_DeepLink.add_argument('--supervised_epochs',   default=2000,         type=int)

    parser_DeepLink.add_argument('--train_dict',          default="dataspace/dictionaries/node,split=0.2.train.dict")
    parser_DeepLink.add_argument('--hidden_dim1',         default=1200, type=int)
    parser_DeepLink.add_argument('--hidden_dim2',         default=1600, type=int)

    parser_DeepLink.add_argument('--number_walks',        default=1000, type=int)
    parser_DeepLink.add_argument('--format',              default="edgelist")
    parser_DeepLink.add_argument('--walk_length',         default=5, type=int)
    parser_DeepLink.add_argument('--window_size',         default=2, type=int)
    parser_DeepLink.add_argument('--top_k',               default=5, type=int)
    parser_DeepLink.add_argument('--alpha',               default=0.8, type=float)
    parser_DeepLink.add_argument('--num_cores',           default=8, type=int)


    parser_HDA = subparsers.add_parser('HDA', help='HDA algorithm')
    parser_HDA.add_argument('--vector_size',              default=10000, type=int)
    parser_HDA.add_argument('--node_features', nargs='+', default=["page_rank", "node_degree", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality", "clustering_coefficient"])

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    source_nodes = source_dataset.G.nodes()
    target_nodes = target_dataset.G.nodes()
    groundtruth_matrix = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx)

    algorithm = args.algorithm

    if (algorithm == "IsoRank"):
        model = IsoRank(source_dataset, target_dataset, args.H, args.alpha, args.max_iter, args.tol)
    elif (algorithm == "FINAL"):
        model = FINAL(source_dataset, target_dataset, H=args.H, alpha=args.alpha, maxiter=args.max_iter, tol=args.tol)
    elif (algorithm == "REGAL"):
        model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
    elif algorithm == "BigAlign":
        model = BigAlign(source_dataset, target_dataset, lamb=args.lamb)
    elif algorithm == "IONE":
        model = IONE(source_dataset, target_dataset, gt_train=args.train_dict, epochs=args.epochs, dim=args.dim, seed=args.seed)
    elif algorithm == "PALE":
        model = PALE(source_dataset, target_dataset, args)
    elif algorithm == "DeepLink":
        model = DeepLink(source_dataset, target_dataset, args)
    elif algorithm == "HDA":
        model = HDA(source_dataset, target_dataset, vector_size=args.vector_size, node_features=args.node_features)
    else:
        raise Exception("Unsupported algorithm")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(7)

    start_time = time()

    S = model.align()
    pred, true_positive_sources = get_statistics(S, groundtruth_matrix)

    # [MY CODE]
    # Extract the true aligned node names:
    # We already have the node index for the source (indices in 'true_positive_sources'),
    # we need to get for each row the target node index, retrieve for both indices
    # the node names and store the result in a tuple.
    true_alignments = []
    for i in true_positive_sources:

        # Get the 'target' node index
        true_row = np.array(pred[i])
        true_target = np.where(true_row == 1)[0].item()

        # Get the node names
        source_name = list(source_dataset.G.nodes())[i]
        target_name = list(target_dataset.G.nodes())[true_target]

        true_alignments.append((source_name, target_name))

    # print("\nTrue alignments:\n", true_alignments, end="\n\n")

    # Get the degrees for each true alignment:
    source_degrees = {}
    target_degrees = {}

    for s, t in true_alignments:
        source_degrees[s] = source_dataset.G.degree(s)
        target_degrees[t] = target_dataset.G.degree(t)

    
    source_degrees = dict(sorted(source_degrees.items(), key=lambda x:x[1], reverse=True))
    target_degrees = dict(sorted(target_degrees.items(), key=lambda x:x[1], reverse=True))

    with open(f"results/ppi_{args.algorithm}_source_degrees.txt", 'w') as f:  
        for key, value in source_degrees.items():  
            f.write('%s:%s\n' % (key, value))

    with open(f"results/ppi_{args.algorithm}_target_degrees.txt", 'w') as f:  
        for key, value in target_degrees.items():  
            f.write('%s:%s\n' % (key, value))


    # Get all degrees
    all_source_degrees = {n: source_dataset.G.degree(n) for n in source_dataset.G.nodes}
    all_target_degrees = {n: target_dataset.G.degree(n) for n in target_dataset.G.nodes}

    all_source_degrees = dict(sorted(all_source_degrees.items(), key=lambda x:x[1], reverse=True))
    all_target_degrees = dict(sorted(all_target_degrees.items(), key=lambda x:x[1], reverse=True))

    with open(f"results/ppi_{args.algorithm}_all_source_degrees.csv", 'w') as f:  
        for key, value in all_source_degrees.items():  
            f.write('%s:%s\n' % (key, value))

    with open(f"results/ppi_{args.algorithm}_all_target_degrees.csv", 'w') as f:  
        for key, value in all_target_degrees.items():  
            f.write('%s:%s\n' % (key, value))


    print("Full_time: ", time() - start_time)









