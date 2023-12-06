from input.semi_synthetic import SemiSynthetic
from input.fully_synthetic import FullySynthetic
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of fully synthetic datasets")
    parser.add_argument('--output_path', default="dataspace/fully-synthetic")
    parser.add_argument('--model', default="small_world", type=str, help="The model with wich generate the dataset. Choose from: 'small_world' and 'erdos_renyi' (default: 'small_world').")
    parser.add_argument('--n', default=10000, type=int, help="Number of nodes")
    parser.add_argument('--aver', default=5, type=float, help="Average degree")
    parser.add_argument('--feature_dim', default=None, type=int, help="Size of node features (default: None).")
    return parser.parse_args()


def gen_fully(full, n_nodes, n_edges, p, feature_dim=None):
    name_p = str(int(p))
    outdir = full+"/erdos_renyi-n{}-p{}".format(n_nodes, name_p)
    FullySynthetic.generate_erdos_renyi_graph(outdir, n_nodes=n_nodes, n_edges=n_edges, feature_dim=feature_dim)
    return outdir


def gen_fully_smallworld(full, n_nodes, p, feature_dim=None):
    name_p = str(int(p))
    outdir = full+"/small_world-n{}-p{}".format(n_nodes,name_p)
    FullySynthetic.generate_small_world_graph(outdir, n_nodes, int(p), 0.35, feature_dim=feature_dim)
    return outdir


if __name__ == "__main__":
    args = parse_args()
    num_edges = args.aver * args.n / 2
    
    if args.model == "erdos_renyi":
        gen_fully(args.output_path, args.n, num_edges, args.aver, feature_dim=args.feature_dim)
    elif args.model == "small_world":
        gen_fully_smallworld(args.output_path, args.n, args.aver, feature_dim=args.feature_dim)
    else:
        raise ValueError(f"{args.model} is not valid. Choose from: 'erdos_renyi' and 'small_world'.")
