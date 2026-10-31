"""Conditional flow matching: prior, loss, training loop and ODE sampling (Klein et al. 2023, eq. 11)."""

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from tqdm.auto import tqdm


def remove_mean(x: torch.Tensor) -> torch.Tensor:
    """x: [B, N, D] -> the same configurations with zero center of mass."""
    return x - x.mean(1, keepdim=True)


def sample_prior(B: int, N: int, D: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Mean-free standard Gaussian: [B, N, D]."""
    sample = torch.randn(B, N, D, device=device, dtype=dtype)
    return sample - sample.mean(dim=1, keepdim=True)


def cfm_loss(model: nn.Module, x1: torch.Tensor, sigma: float = 0.01, coupling=None) -> torch.Tensor:
    """
    x1: [B, N, D] batch of centered data samples.
    Draws x0 ~ q and t ~ U[0, 1], builds x_t, returns the scalar CFM loss.
    coupling: None (independent pairs) or a function (x0, x1) -> (x0, x1) that matches the pairs, e.g. OT.
    """
    B, N, D = x1.shape
    x0 = sample_prior(B, N, D, device=x1.device, dtype=x1.dtype)
    if coupling is not None:
        x0, x1 = coupling(x0, x1)
    eps = sample_prior(B, N, D, device=x1.device, dtype=x1.dtype)
    t = torch.rand(B, 1, 1, device=x1.device, dtype=x1.dtype)

    x_t = t * x1 + (1 - t) * x0 + sigma * eps

    v = model(t.squeeze(-1), x_t)

    loss = ((v - (x1 - x0))**2).sum(dim=(1, 2)).mean()

    return loss


@torch.no_grad()
def val_loss(model, loader, sigma, seed=0, coupling=None):
    """CFM loss on held-out data with fixed x0, t, eps (fixed seed), so epochs are comparable."""
    device = next(model.parameters()).device
    model.eval()
    # fork_rng saves the CPU (and CUDA) RNG state and restores it on exit: validation does not disturb training randomness
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        loss = sum(cfm_loss(model, batch.to(device), sigma=sigma, coupling=coupling).item() * len(batch) for batch in loader) / len(loader.dataset)
    model.train()
    return loss


def train(model, x_train, x_val, n_epochs, lr=5e-4, lr_milestones=None, batch_size=256, sigma=0.01, coupling=None):
    """
    Trains the vector field with the CFM loss. LR is multiplied by 0.1 after each epoch in lr_milestones
    (default: half of training; the paper uses 5e-4 -> 5e-5 after 200 of 400 epochs).
    coupling is used both for training and validation: it defines the regression targets.
    Returns per-step train losses and per-epoch validation losses.
    """
    device = next(model.parameters()).device
    if lr_milestones is None:
        lr_milestones = [n_epochs // 2]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size)
    steps_per_epoch = len(train_loader)

    losses, val_losses = [], []

    for epoch in (pbar := tqdm(range(n_epochs))):
        for batch in train_loader:
            loss = cfm_loss(model, x1=batch.to(device), sigma=sigma, coupling=coupling)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()   # once per epoch: milestones are in epochs
        val_losses.append(val_loss(model, val_loader, sigma, coupling=coupling))
        pbar.set_postfix(train=f"{np.mean(losses[-steps_per_epoch:]):.3f}", val=f"{val_losses[-1]:.3f}",
                         lr=scheduler.get_last_lr()[0])

    return losses, val_losses


class torch_wrapper(torch.nn.Module):
    """Adapts the vector field to torchdiffeq: odeint calls f(t, x) with a scalar t, the model expects t of shape [B]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(t.repeat(x.shape[0]), x)


@torch.no_grad()
def integrate(model: nn.Module, x0: torch.Tensor, n_steps: int = 100, method: str = "euler") -> torch.Tensor:
    """x0: [B, N, D] prior samples -> x1: [B, N, D] model samples. method: "euler" or "rk4"."""
    x = odeint(func=torch_wrapper(model), y0=x0, t=torch.linspace(0, 1, n_steps + 1, device=x0.device), method=method)
    x1 = x[-1].cpu()
    return x1


@torch.no_grad()
def integrate_dopri5(model: nn.Module, x0: torch.Tensor, atol=1e-5, rtol=1e-5) -> torch.Tensor:
    """Adaptive reference solver via torchdiffeq."""
    x = odeint(func=torch_wrapper(model), y0=x0, t=torch.tensor([0., 1.], device=x0.device), method="dopri5", atol=atol, rtol=rtol)
    x1 = x[-1].cpu()
    return x1
