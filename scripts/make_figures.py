"""Collect results/*/metrics.json into figures and a summary table.

Writes results/fig_distributions.png, results/fig_nfe.png, results/summary.md.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eqfm.dw4 import dw4_energy, load_dw4
from eqfm.plotting import pairwise_distances

RESULTS = "results"
ARCHS = [("egnn", "EGNN (equivariant)", "-"), ("mlp", "MLP (not equivariant)", "--")]
COUPLINGS = [("independent", "independent", "#2a78d6"), ("ot", "OT", "#eb6834"), ("eq_ot", "equivariant OT", "#1baf7a")]
DATA_COLOR, INK, MUTED = "#b5b4ad", "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False})


def load_runs():
    runs = {}
    for arch, *_ in ARCHS:
        for coupling, *_ in COUPLINGS:
            path = os.path.join(RESULTS, f"{arch}_{coupling}")
            if os.path.exists(os.path.join(path, "metrics.json")):
                with open(os.path.join(path, "metrics.json")) as f:
                    metrics = json.load(f)
                runs[arch, coupling] = (metrics, torch.from_numpy(np.load(os.path.join(path, "samples.npy"))))
    return runs


def fig_distributions(runs, x_ref):
    e_ref, d_ref = dw4_energy(x_ref).numpy(), pairwise_distances(x_ref).numpy()
    e_lo, e_hi = np.percentile(e_ref, 0.1), np.percentile(e_ref, 99.9) + 5
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex="col")
    for row, (arch, arch_label, _) in enumerate(ARCHS):
        ax_d, ax_e = axes[row]
        ax_d.hist(d_ref, bins=100, range=(0, 8), density=True, color=DATA_COLOR, label="data (MCMC)")
        ax_e.hist(e_ref, bins=100, range=(e_lo, e_hi), density=True, color=DATA_COLOR, label="data (MCMC)")
        for coupling, label, color in COUPLINGS:
            if (arch, coupling) not in runs:
                continue
            samples = runs[arch, coupling][1]
            ax_d.hist(pairwise_distances(samples).numpy(), bins=100, range=(0, 8), density=True,
                      histtype="step", lw=1.5, color=color, label=label)
            ax_e.hist(np.clip(dw4_energy(samples).numpy(), e_lo, e_hi), bins=100, range=(e_lo, e_hi), density=True,
                      histtype="step", lw=1.5, color=color, label=label)
        ax_d.set_ylabel(f"{arch_label}\ndensity")
        ax_d.set_title("interparticle distance $d_{ij}$" if row == 0 else "", loc="left")
        ax_e.set_title("energy $U(x)$ (right tail clipped into last bin)" if row == 0 else "", loc="left")
    axes[1, 0].set_xlabel("$d_{ij}$")
    axes[1, 1].set_xlabel("$U(x)$")
    axes[0, 1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_distributions.png"), dpi=200)


def fig_nfe(runs):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for arch, arch_label, ls in ARCHS:
        for coupling, label, color in COUPLINGS:
            if (arch, coupling) not in runs:
                continue
            w1 = runs[arch, coupling][0]["euler_energy_w1"]
            steps = sorted(int(n) for n in w1)
            ax.plot(steps, [w1[str(n)] for n in steps], ls=ls, lw=2, marker="o", ms=4, color=color,
                    label=f"{arch.upper()} + {label}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Euler steps (= function evaluations)")
    ax.set_ylabel("$W_1$(energy of samples, energy of data)")
    ax.set_title("Sample quality vs. number of integration steps (lower is better)", loc="left")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_nfe.png"), dpi=200)


def summary_table(runs):
    lines = ["| architecture | coupling | params | energy $W_1$ (dopri5) | $d_{ij}$ $W_1$ | path length | "
             "dopri5 NFE | energy $W_1$, Euler 4 steps | energy $W_1$, Euler 16 steps | equivariance error | train, min |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arch, arch_label, _ in ARCHS:
        for coupling, label, _ in COUPLINGS:
            if (arch, coupling) not in runs:
                continue
            m = runs[arch, coupling][0]
            lines.append(f"| {arch.upper()} | {label} | {m['n_params']} | {m['energy_w1']:.3f} | {m['distance_w1']:.3f} | "
                         f"{m['path_length']:.2f} | {m['dopri5_nfe']} | {m['euler_energy_w1']['4']:.2f} | "
                         f"{m['euler_energy_w1']['16']:.3f} | {m['equivariance_error']:.1e} | {m['train_minutes']:.0f} |")
    table = "\n".join(lines)
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)


def main():
    runs = load_runs()
    if not runs:
        print("no finished runs in results/")
        return
    _, x_test = load_dw4(n_test=10_000)
    fig_distributions(runs, x_test)
    fig_nfe(runs)
    summary_table(runs)


if __name__ == "__main__":
    main()
