"""Train one configuration (architecture x coupling) on DW4 and evaluate it.

    python scripts/run_experiment.py --arch egnn --coupling eq_ot

Writes results/<arch>_<coupling>/: model.pt, losses.json, metrics.json, samples.npy.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

from eqfm.couplings import ot_coupling
from eqfm.dw4 import dw4_energy, load_dw4
from eqfm.egnn import EGNNVectorField
from eqfm.flow_matching import cfm_loss, integrate, integrate_dopri5, sample_prior, train
from eqfm.mlp import MLPVectorField
from eqfm.plotting import pairwise_distances

EULER_STEPS = [1, 2, 4, 8, 16, 32, 64, 128]


def train_precomputed(model, pairs, n_epochs, lr, lr_milestones, sigma):
    """Same loop as flow_matching.train, but over precomputed equivariant-OT pairs (one pair set per epoch).

    Batches keep their composition (OT matching is done within a batch); only the batch order is shuffled.
    The stored pairs are fed to the unchanged cfm_loss through a `coupling` that returns them.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    n_sets, n_train = pairs["train_x0"].shape[:2]
    batch_size = pairs["batch_size"]
    starts = torch.arange(0, n_train, batch_size)

    def batches(x0_all, x1_all, order):
        for s in order:
            yield x0_all[s:s + batch_size].to(device), x1_all[s:s + batch_size].to(device)

    def loss_on(x0, x1):
        return cfm_loss(model, x1, sigma=sigma, coupling=lambda _x0, _x1: (x0, _x1))

    losses, val_losses = [], []
    for epoch in (pbar := tqdm(range(n_epochs), mininterval=30)):
        k = epoch % n_sets
        for x0, x1 in batches(pairs["train_x0"][k], pairs["train_x1"][k], starts[torch.randperm(len(starts))]):
            loss = loss_on(x0, x1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(0)
            n_val = len(pairs["val_x0"])
            val = sum(loss_on(x0, x1).item() * len(x0)
                      for x0, x1 in batches(pairs["val_x0"], pairs["val_x1"], range(0, n_val, batch_size))) / n_val
        model.train()
        val_losses.append(val)
        pbar.set_postfix(train=f"{np.mean(losses[-len(starts):]):.3f}", val=f"{val:.3f}")
    return losses, val_losses


class CountingField(nn.Module):
    """Wraps a vector field and counts its evaluations (NFE)."""

    def __init__(self, model):
        super().__init__()
        self.model, self.n_calls = model, 0

    def forward(self, t, x):
        self.n_calls += 1
        return self.model(t, x)


def random_group_element(D, N, generator):
    """Random element of O(D) x S(N): an orthogonal matrix (rotation or reflection) and a permutation."""
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=generator))
    return Q, torch.randperm(N, generator=generator)


@torch.no_grad()
def evaluate(model, x_ref, n_eval, seed=1234):
    """Sample with dopri5 and fixed-step Euler from the same prior noise; compare with reference data."""
    device = next(model.parameters()).device
    model.eval()
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        x0 = sample_prior(n_eval, *x_ref.shape[1:], device=device)
    x_ref = x_ref[:n_eval]
    e_ref, d_ref = dw4_energy(x_ref).numpy(), pairwise_distances(x_ref).numpy()

    counting = CountingField(model)
    start = time.time()
    x1 = integrate_dopri5(counting, x0)
    metrics = {
        "dopri5_nfe": counting.n_calls,
        "dopri5_seconds": time.time() - start,
        "energy_w1": wasserstein_distance(dw4_energy(x1).numpy(), e_ref),
        "distance_w1": wasserstein_distance(pairwise_distances(x1).numpy(), d_ref),
        "path_length": (x1 - x0.cpu()).flatten(1).norm(dim=1).mean().item(),   # ||x(1) - x(0)||, as in Table 1
        "mean_energy": dw4_energy(x1).mean().item(),
        "mean_energy_data": float(e_ref.mean()),
    }

    metrics["euler_energy_w1"] = {}
    for n in EULER_STEPS:
        x1_n = integrate(model, x0, n_steps=n, method="euler")
        metrics["euler_energy_w1"][n] = wasserstein_distance(dw4_energy(x1_n).numpy(), e_ref)

    # equivariance of the learned flow map F: x0 -> x1 (Euler, 32 steps): ||F(g x0) - g F(x0)|| / ||F(x0)||
    generator = torch.Generator().manual_seed(seed)
    m = min(1000, n_eval)
    R, perm = random_group_element(x0.shape[2], x0.shape[1], generator)
    R = R.to(device)
    F = integrate(model, x0[:m], n_steps=32).to(device)
    F_g = integrate(model, (x0[:m] @ R.T)[:, perm], n_steps=32).to(device)
    metrics["equivariance_error"] = ((F_g - (F @ R.T)[:, perm]).norm() / F.norm()).item()

    model.train()
    return metrics, x1


def build_model(arch, N, D):
    if arch == "egnn":
        return EGNNVectorField(hidden_dim=32, n_layers=3)       # Table 7 of the paper for DW4
    return MLPVectorField(N, D)                                 # parameter count matched to the EGNN (~25k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["egnn", "mlp"], required=True)
    parser.add_argument("--coupling", choices=["independent", "ot", "eq_ot"], required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=10_000)
    parser.add_argument("--n-eval", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", default="data/dw4_eq_ot_pairs.pt")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    name = f"{args.arch}_{args.coupling}"
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    x_train, x_test = load_dw4(n_train=args.n_train, n_test=max(args.n_val, args.n_eval))
    N, D = x_train.shape[1:]
    model = build_model(args.arch, N, D)
    n_params = sum(p.numel() for p in model.parameters())
    lr_milestones = [args.epochs // 2]                           # 5e-4 -> 5e-5 halfway, as in Table 8
    print(f"{name}: {n_params} parameters, {args.epochs} epochs")

    start = time.time()
    if args.coupling == "eq_ot":
        pairs = torch.load(args.pairs)
        losses, val_losses = train_precomputed(model, pairs, args.epochs, args.lr, lr_milestones, args.sigma)
    else:
        coupling = ot_coupling if args.coupling == "ot" else None
        losses, val_losses = train(model, x_train, x_test[:args.n_val], args.epochs, lr=args.lr,
                                   lr_milestones=lr_milestones, sigma=args.sigma, coupling=coupling)
    train_minutes = (time.time() - start) / 60

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "losses.json"), "w") as f:
        json.dump({"train": losses[::10], "val": val_losses}, f)

    metrics, samples = evaluate(model, x_test, args.n_eval)
    metrics.update({"arch": args.arch, "coupling": args.coupling, "n_params": n_params, "epochs": args.epochs,
                    "n_train": args.n_train, "train_minutes": train_minutes, "final_val_loss": val_losses[-1]})
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(out_dir, "samples.npy"), samples.numpy())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
