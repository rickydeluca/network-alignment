import argparse
import random
from time import time

import numpy as np
import torch

import utils.graph_utils as graph_utils
from algorithms import (
    COMMON,
    FINAL,
    HDA,
    IONE,
    MAGNA,
    PALE,
    REGAL,
    SANE,
    BigAlign,
    DeepLink,
    IsoRank,
    SHELLEY
)
from evaluation.metrics import get_statistics
from input.dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="dataspace/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="dataspace/douban/offline/graphsage/")
    parser.add_argument('--groundtruth',    default="dataspace/douban/dictionaries/groundtruth")
    parser.add_argument('--alignment_matrix_name', default=None, help="Prefered name of alignment matrix.")
    parser.add_argument('--transpose_alignment_matrix', action="store_true", default=False, help="Transpose the alignment matrix.")
    parser.add_argument('--seed',           default=123,    type=int)

    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: IsoRank, FINAL, UniAlign, PALE, DeepLink, REGAL, IONE, HDA, MAGNA, SANE, COMMON, SHELLEY')

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

    parser_MAGNA = subparsers.add_parser('MAGNA', help='MAGNA++ algorithm')
    parser_MAGNA.add_argument('--source_edgelist', default=None, type=str)
    parser_MAGNA.add_argument('--target_edgelist', default=None, type=str)
    parser_MAGNA.add_argument('--measure', default="S3", type=str)
    parser_MAGNA.add_argument('--population_size', default=15000, type=int)
    parser_MAGNA.add_argument('--num_generations', default=2000, type=int)
    parser_MAGNA.add_argument('--num_threads', default=8, type=int)
    parser_MAGNA.add_argument('--outfile', default="algorithms/MAGNA/output/magna", type=str)
    parser_MAGNA.add_argument('--reverse', action="store_true", default=False)
    
    parser_SANE = subparsers.add_parser('SANE', help='SANE algorithm')
    parser_SANE.add_argument('--train_dict', type=str, default="dataspace/douban/dictionaries/node,split=0.2.train.dict", help="Path to the alignment dictionary for training.")
    parser_SANE.add_argument('--device', type=str, default="cpu", help="Device to use for training (default: cpu)")
    
    # Unsupervised learning parameters
    parser_SANE.add_argument('--embedding_model', type=str, default='sage', help="Model used to learn embeddings (default: sage).")
    parser_SANE.add_argument('--embedding_dim', type=int, default=1024, help="Size for the learned embeddings (default: 1024).")
    parser_SANE.add_argument('--num_layers', type=int, default=1, help="Num layers for the embedding model (default: 1).")
    parser_SANE.add_argument('--hidden_channels', type=int, default=64, help="Size of hidden layers for embedding model (default: 64).")
    parser_SANE.add_argument('--dropout_emb', type=float, default=0.2, help="Dropout probability of the embedding model (default: 0.2).")
    parser_SANE.add_argument('--pos_info', action='store_true', default=True, help="If True concatenate the node features with the (weighted) adjacency matrix (default: True).")
    
    parser_SANE.add_argument('--prediction_model', type=str, default='inner_product', help="Model to use to perform link prediction during unsupervised learning (it can have learnable parameters) (default: inner_product).")
    parser_SANE.add_argument('--dropout_pred', type=float, default=0.5, help="Dropout probability of prediction model, used only if required (default: 64).")
    
    parser_SANE.add_argument('--batch_size_emb', type=int, default=256, help="Batch size for unsupervised learning (default: 256).")
    parser_SANE.add_argument('--epochs_emb', type=int, default=100, help="Number of epoch of unsupervised learning (default: 100).")
    parser_SANE.add_argument('--lr_emb', type=float, default=0.0003, help="Learning rate for unsupervised learning (default: 0.0003).")
    parser_SANE.add_argument('--early_stop_emb', action="store_true", default=False, help="Unsupervised training with early stopping (default: False)")
    parser_SANE.add_argument('--patience_emb', type=int, default=10, help="Number of epochs to wait for improvement if using `early_stop_emb` (default: 10)")

    # Supervised learning parmeters
    parser_SANE.add_argument('--mapping_model', type=str, default='linear', help="Model used to learn mapping (default: linear).")
    parser_SANE.add_argument('--heads', type=int, default=1, help="Number of heads for cross attention. Used only if 'cross_transformer' was chosen as `mapping_model (deafult: 1).")

    parser_SANE.add_argument('--batch_size_map', type=int, default=256, help="Batch size for supervised learning (default: 256).")
    parser_SANE.add_argument('--epochs_map', type=int, default=100, help="Number of epoch of supervised learning (default: 100).")
    parser_SANE.add_argument('--lr_map', type=float, default=0.01, help="Learning rate for supervised learning (default: 0.01).")
    parser_SANE.add_argument('--early_stop_map', action="store_true", default=False, help="Supervised training with early stopping (default: False)")
    parser_SANE.add_argument('--patience_map', type=int, default=10, help="Number of epochs to wait for improvement if using `early_stop_map` (default: 10)")

    # COMMON
    parser_COMMON = subparsers.add_parser('COMMON', help='COMMON algorithm')
    parser_COMMON.add_argument('--cuda', action="store_true")
    parser_COMMON.add_argument('--train_dict', default='dataspace/ppi/dictionaries/node,split=0.2.train.dict')
    
    parser_COMMON.add_argument('--emb_batch_size', type=int, default=512)
    parser_COMMON.add_argument('--emb_epochs', type=int, default=1000)
    parser_COMMON.add_argument('--emb_lr', type=float, default=0.01)
    parser_COMMON.add_argument('--neg_sample_size', type=int, default=512)
    parser_COMMON.add_argument('--embedding_dim', type=int, default=1024)
    parser_COMMON.add_argument('--embedding_name', type=str, default='')

    parser_COMMON.add_argument('--map_batch_size', type=int, default=32)
    parser_COMMON.add_argument('--map_epochs', type=int, default=20)
    parser_COMMON.add_argument('--map_epoch_iters', type=int, default=2000)
    parser_COMMON.add_argument('--map_lr', type=float, default=3e-4)
    parser_COMMON.add_argument('--map_lr_decay', type=float, default=0.5)
    parser_COMMON.add_argument('--map_lr_step', type=list, nargs='+', default=[2,4,6,8,10])
    parser_COMMON.add_argument('--map_loss_func', type=str, default='custom')
    parser_COMMON.add_argument('--map_optimizer', type=str, default='adam')
    parser_COMMON.add_argument('--backbone', type=str, default='splinecnn')
    parser_COMMON.add_argument('--rescale', type=int, default=256)
    parser_COMMON.add_argument('--separate_backbone_lr', action='store_true', default=False)
    parser_COMMON.add_argument('--backbone_lr', type=float, default=3e-4)
    parser_COMMON.add_argument('--map_optimizer_momentum', type=float, default=0.9)
    parser_COMMON.add_argument('--map_softmax_temp', type=float, default=0.07)
    parser_COMMON.add_argument('--map_eval_epochs', type=int, default=20)
    parser_COMMON.add_argument('--map_feature_channel', type=int, default=512)
    parser_COMMON.add_argument('--alpha', type=float, default=0.4)
    parser_COMMON.add_argument('--distill', action='store_true', default=True)
    parser_COMMON.add_argument('--warmup_step', type=int, default=2000)
    parser_COMMON.add_argument('--distill_momentum', type=float, default=0.995)
    
    # SHELLEY
    parser_SHELLEY = subparsers.add_parser('SHELLEY', help='SHELLEY algorithm')
    parser_SHELLEY.add_argument('--train_dict', default='dataspace/ppi/dictionaries/node,split=0.2.train.dict')
    parser_SHELLEY.add_argument('--use_pretrained', action='store_true', default=False)
    parser_SHELLEY.add_argument('--eval', action='store_true', default=True)
    parser_SHELLEY.add_argument('--seed', type=int, default=42)
    parser_SHELLEY.add_argument('--cuda', action="store_true", default=False)
    
    parser_SHELLEY.add_argument('--root_dir', type=str, default='dataspace/ppi')
    parser_SHELLEY.add_argument('--p_add', type=float, default=0.1)
    parser_SHELLEY.add_argument('--p_rm', type=float, default=0.1)

    parser_SHELLEY.add_argument('--backbone', type=str, default='gin')
    parser_SHELLEY.add_argument('--node_feature_dim', type=int, default=1)
    parser_SHELLEY.add_argument('--embedding_dim', type=int, default=256)
    parser_SHELLEY.add_argument('--softmax_temp', type=float, default=0.07)
    
    parser_SHELLEY.add_argument('--head', type=str, default='common')
    parser_SHELLEY.add_argument('--distill', action='store_true', default=True)
    parser_SHELLEY.add_argument('--distill_momentum', type=float, default=0.995)
    parser_SHELLEY.add_argument('--warmup_step', type=int, default=-1)  # if -1 set it equal to `epoch_iters`
    parser_SHELLEY.add_argument('--alpha', type=float, default=0.4)

    parser_SHELLEY.add_argument('--feature_channel', type=int, default=512)
    parser_SHELLEY.add_argument('--sk_iter_num', type=int, default=1024)
    parser_SHELLEY.add_argument('--sk_epsilon', type=float, default=1e-10)
    parser_SHELLEY.add_argument('--sk_tau', type=float, default=0.003)

    parser_SHELLEY.add_argument('--train', action='store_true', default=False)
    parser_SHELLEY.add_argument('--self_supervised', action='store_true', default=False)
    parser_SHELLEY.add_argument('--optimizer', type=str, default='adam')
    parser_SHELLEY.add_argument('--loss_func', type=str, default='distill_qc')
    parser_SHELLEY.add_argument('--optim_momentum', type=float, default=0.9)
    parser_SHELLEY.add_argument('--lr', type=float, default=3e-4)
    parser_SHELLEY.add_argument('--use_scheduler', action='store_true', default=False)
    parser_SHELLEY.add_argument('--lr_step', type=list, nargs='+', default=[2,4,6,8,10])
    parser_SHELLEY.add_argument('--lr_decay', type=float, default=0.5)
    parser_SHELLEY.add_argument('--separate_backbone_lr', action='store_true', default=False)
    parser_SHELLEY.add_argument('--backbone_lr', type=float, default=2e-5)
    parser_SHELLEY.add_argument('--start_epoch', type=int, default=0)
    parser_SHELLEY.add_argument('--num_epochs', type=int, default=20)
    parser_SHELLEY.add_argument('--epoch_iters', type=int, default=-1)  # if -1 use the whole training dataset
    parser_SHELLEY.add_argument('--batchsize', type=int, default=4)     # if '-1' do not perform minibatching
    parser_SHELLEY.add_argument('--statistic_step', type=int, default=4)
    parser_SHELLEY.add_argument('--early_stopping', action='store_true', default=True)
    parser_SHELLEY.add_argument('--patience', type=int, default=5)
    parser_SHELLEY.add_argument('--checkpoints', type=str, default='algorithms/SHELLEY/checkpoints')
    
    parser_SHELLEY.add_argument('--validate', action='store_true', default=False)
    parser_SHELLEY.add_argument('--val_iters', type=int, default=100)

    parser_SHELLEY.add_argument('--test', action='store_true', default=False)
    parser_SHELLEY.add_argument('--test_iters', type=int, default=100)

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
    elif algorithm == "MAGNA":
        model = MAGNA(source_dataset, target_dataset, source_edgelist=args.source_edgelist, target_edgelist=args.target_edgelist, measure=args.measure, population_size=args.population_size, num_generations=args.num_generations, num_threads=args.num_threads, outfile=args.outfile, reverse=args.reverse)
    elif algorithm == "SANE":
        model = SANE(source_dataset, target_dataset, args)
    elif algorithm == "COMMON":
        model = COMMON(source_dataset, target_dataset, args)
    elif algorithm == "SHELLEY":
        model = SHELLEY(source_dataset, target_dataset, args)
    else:
        raise Exception("Unsupported algorithm")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_time = time()

    S = model.align()
    get_statistics(S, groundtruth_matrix)

    print(f"Full_time: {time() - start_time}")