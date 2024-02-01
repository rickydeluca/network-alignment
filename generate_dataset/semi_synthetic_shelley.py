import argparse

from input.semi_synthetic import SemiSynthetic


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a noisy graph from a source graph.")
    parser.add_argument('--input_path', default="data/ppi", type=str)
    parser.add_argument('--idx', default=1, type=int)
    parser.add_argument('--p_add', default=0.1, type=float)
    parser.add_argument('--p_rm', default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weighted', action="store_true")
    parser.add_argument('--outdir', default=None, type=str)
    return parser.parse_args()

def gen_SHELLEY(input_path,
                idx=1,
                p_add=0.1,
                p_rm=0.1,
                seed=42,
                weighted=False,
                outdir=None):
    
    networkx_dir = input_path + '/graphsage'

    if outdir is None:
        outdir = input_path + f'targets/{idx}_add{p_add}_rm{p_rm}/'

    semiSynthetic = SemiSynthetic(networkx_dir, outdir, seed=seed, weighted=weighted)
    semiSynthetic.generate_random_clone_synthetic_shelley(p_add=p_add,
                                                          p_rm=p_rm)
    

if __name__ == "__main__":
    args = parse_args()
    gen_SHELLEY(input_path=args.input_path,
                idx=args.idx,
                p_add=args.p_add,
                p_rm=args.p_rm,
                seed=args.seed,
                weighted=args.weighted,
                outdir=args.outdir)
    print("Done!")

