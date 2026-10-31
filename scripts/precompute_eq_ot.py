"""Precompute equivariant-OT pairs (x0, aligned x1) for training and validation.

Equivariant OT costs ~3.6 s per batch of 256 on one CPU core (B^2 Hungarian + Kabsch alignments), which is far
too slow to run inside the training loop. As in the paper (App. C.4), the pairs are computed beforehand:
K independent pair sets, each one a random partition of the training set into batches with fresh prior noise,
matched batch by batch. Training cycles through the K sets, one per epoch. Batches run in parallel processes.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eqfm.couplings import eq_ot_coupling_bruteforce as eq_ot_coupling   # exact for DW4; verified against the loop version in tests
from eqfm.dw4 import load_dw4
from eqfm.flow_matching import sample_prior


def _match(args):
    x0, x1 = args
    torch.set_num_threads(1)
    return eq_ot_coupling(x0, x1)


def make_pairs(x, n_sets, batch_size, seed, pool):
    generator = torch.Generator().manual_seed(seed)
    B, N, D = x.shape
    jobs = []
    for _ in range(n_sets):
        perm = torch.randperm(B, generator=generator)
        for idx in perm.split(batch_size):
            x0 = torch.randn(len(idx), N, D, generator=generator)
            jobs.append((x0 - x0.mean(dim=1, keepdim=True), x[idx]))
    results = pool.map(_match, jobs, chunksize=1)
    x0 = torch.cat([r[0] for r in results]).reshape(n_sets, B, N, D)
    x1 = torch.cat([r[1] for r in results]).reshape(n_sets, B, N, D)
    return x0, x1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=10_000)
    parser.add_argument("--n-sets", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="data/dw4_eq_ot_pairs.pt")
    args = parser.parse_args()

    if os.path.exists(args.out):
        saved = torch.load(args.out)
        if saved["train_x0"].shape[:2] == (args.n_sets, args.n_train) and len(saved["val_x0"]) == args.n_val:
            print(f"{args.out} already exists with the same settings, skipping")
            return

    x_train, x_test = load_dw4(n_train=args.n_train, n_test=args.n_val)
    start = time.time()
    with Pool(args.workers) as pool:
        train_x0, train_x1 = make_pairs(x_train, args.n_sets, args.batch_size, seed=0, pool=pool)
        val_x0, val_x1 = make_pairs(x_test, 1, args.batch_size, seed=1, pool=pool)
    torch.save({"train_x0": train_x0, "train_x1": train_x1, "val_x0": val_x0[0], "val_x1": val_x1[0],
                "batch_size": args.batch_size}, args.out)
    cost = ((train_x0 - train_x1) ** 2).sum(dim=(-1, -2)).mean().item()
    print(f"saved {args.out}: {args.n_sets} x {args.n_train} pairs, mean cost {cost:.2f}, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
