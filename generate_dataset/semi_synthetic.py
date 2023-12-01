from input.semi_synthetic import SemiSynthetic
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Erdos Renyi Graph Generation")
    parser.add_argument('--input_path', default="data/ppi")
    parser.add_argument('--d', default=0.05, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--weighted', action="store_true", help="Weighted edgelist")
    return parser.parse_args()

def gen_REGAL(d, input_path, p_change_feats=None, seed=1, weighted=False):
    name_d = str(d).replace("0.","")
    networkx_dir = input_path+'/graphsage'
    if p_change_feats is not None:
        outdir = input_path+'/REGAL-d{}-pfeats{}-seed{}/'.format(name_d,
                                                                  str(p_change_feats).replace("0.",""), seed)
    else:
        outdir = input_path + '/REGAL-d{}-seed{}/'.format(name_d, seed)
    semiSynthetic = SemiSynthetic(networkx_dir, outdir, seed=seed, weighted=weighted)
    semiSynthetic.generate_random_clone_synthetic(0, d, p_change_feats=p_change_feats)

if __name__ == "__main__":
    args = parse_args()
    gen_REGAL(args.d, args.input_path, seed=args.seed, weighted=args.weighted)
    print("Done!")

