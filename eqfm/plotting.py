"""Plots: training curves and model samples vs. data (pair distances and energies)."""

import numpy as np
import torch
import matplotlib.pyplot as plt


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """x: [B, N, D] -> all d_ij with i < j, flattened."""
    N = x.size(1)
    x_i = x.unsqueeze(2).expand(-1, -1, N, -1)                                  # [B, N, N, D]
    x_j = x.unsqueeze(1).expand(-1, N, -1, -1)                                  # [B, N, N, D]

    x_dif = x_i - x_j                                                           # [B, N, N, D]
    d = (x_dif ** 2).sum(-1) ** 0.5                                             # [B, N, N]

    # each pair once: keep only the upper triangle i < j (the matrix is symmetric, the diagonal is 0)
    upper = torch.ones(N, N, device=x.device).triu(diagonal=1).bool()           # [N, N]
    return d[:, upper].flatten()                                                # [B * N(N-1)/2]


def plot_losses(losses, val_losses, steps_per_epoch, baseline_loss=None):
    plt.figure(figsize=(7, 3))
    plt.plot(losses, lw=0.3, alpha=0.5, label="train (per step)")
    plt.plot(steps_per_epoch * np.arange(1, len(val_losses) + 1), val_losses, "o-", ms=3, label="val (per epoch)")
    if baseline_loss is not None:
        plt.axhline(baseline_loss, color="k", ls="--", lw=1, label="v = 0 baseline")
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("CFM loss"); plt.legend()
    plt.show()


def plot_samples_vs_data(samples: torch.Tensor, data: torch.Tensor, energy_fn, label: str = "model"):
    samples, data = samples.detach().cpu(), data.detach().cpu()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(pairwise_distances(data).numpy(), bins=100, density=True, alpha=0.5, label="data")
    ax[0].hist(pairwise_distances(samples).numpy(), bins=100, density=True, alpha=0.5, label=label)
    ax[0].set_xlabel(r"$d_{ij}$"); ax[0].legend()
    e_data, e_model = energy_fn(data).numpy(), energy_fn(samples).numpy()
    lo, hi = np.percentile(e_data, 0.1), np.percentile(e_data, 99.9) + 10
    bins = np.linspace(lo, hi, 100)
    ax[1].hist(e_data, bins=bins, density=True, alpha=0.5, label="data")
    ax[1].hist(np.clip(e_model, lo, hi), bins=bins, density=True, alpha=0.5, label=label)
    ax[1].set_xlabel(r"$U(x)$"); ax[1].legend()
    plt.tight_layout(); plt.show()
