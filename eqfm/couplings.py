"""Matching noise and data inside a batch: minibatch OT (§3.4) and equivariant OT (§4, eq. 16) of Klein et al. 2023."""

import itertools

import torch
from scipy.optimize import linear_sum_assignment


def distance(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """ d[i, j] = ||x0[i] - x1[j]||^2 (squared) """
    B, N, D = x0.shape
    x_0 = x0.unsqueeze(0).expand(B, -1, -1, -1)                                     # [B, B, N, D]
    x_1 = x1.unsqueeze(1).expand(-1, B, -1, -1)                                     # [B, B, N, D]

    x_dif = x_0 - x_1                                                               # [B, B, N, D]

    d = (x_dif ** 2).sum(dim=(2, 3)).transpose(0, 1)                                # [B, B]
    return d


@torch.no_grad()
def ot_coupling(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x0, x1: [B, N, D] -> (x0, x1 reordered along the batch) according to minibatch OT."""
    d = distance(x0, x1)                               # [B, B]

    row_ind, col_ind = linear_sum_assignment(cost_matrix=d.cpu().numpy())
    return x0, x1[col_ind]


def align_permutation(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """x0, x1: [N, D] -> x1 with its particles reordered to best match x0."""
    d = distance(x0.unsqueeze(1), x1.unsqueeze(1))
    row_ind, col_ind = linear_sum_assignment(cost_matrix=d.cpu().numpy())
    return x1[col_ind]


def kabsch(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """x0, x1: [N, D], both centered -> x1 R^T with R in SO(D) minimizing ||x0 - x1 R^T||^2."""
    _, D = x1.shape
    A = x1.transpose(-2, -1) @ x0
    U, S, Vh = torch.linalg.svd(A)
    d = torch.linalg.det(Vh @ U)
    on_diag = [1 for _ in range(D - 1)]
    on_diag.append(1 if d >= 0 else -1)

    diag = torch.diag(torch.tensor(on_diag, device=x1.device, dtype=x1.dtype))
    R_T = U @ diag @ Vh

    return x1 @ R_T


def align(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """x0, x1: [N, D] -> x1 aligned to x0: permutation, then rotation (eq. 16)."""
    x1 = align_permutation(x0, x1)
    x1 = kabsch(x0, x1)
    return x1


@torch.no_grad()
def eq_ot_coupling(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x0, x1: [B, N, D] -> (x0, aligned and reordered x1) according to equivariant OT."""
    B, N, D = x0.shape
    cost_matrix = torch.zeros((B, B), device=x0.device, dtype=x0.dtype)
    for i in range(B):
        for j in range(B):
            x1_aligned = align(x0[i], x1[j])
            cost_matrix[i, j] = ((x0[i] - x1_aligned) ** 2).sum()

    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix.cpu().numpy())

    x1_OT = x1[col_ind]

    x1_OT_aligned = torch.zeros_like(x1_OT)

    for i in range(B):
        x1_OT_aligned[i] = align(x0[i], x1_OT[i])

    return x0, x1_OT_aligned


@torch.no_grad()
def eq_ot_coupling_bruteforce(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same result as eq_ot_coupling, vectorized for small N (used for DW4, N = 4).

    The best particle permutation of every pair is found by trying all N! permutations for all B^2 pairs at once;
    this is the exact optimum, the same one the Hungarian algorithm returns. Kabsch is applied to all pairs in one
    batched SVD. ~50x faster than the loop version for N = 4; infeasible for large N (13! permutations for LJ13).
    """
    B, N, D = x0.shape
    perms = torch.tensor(list(itertools.permutations(range(N))), device=x0.device)       # [N!, N]
    X0 = x0.unsqueeze(1).expand(B, B, N, D)                                              # X0[a, b] = x0[a]
    X1 = x1.unsqueeze(0).expand(B, B, N, D)                                              # X1[a, b] = x1[b]

    perm_cost = ((X0.unsqueeze(2) - X1[:, :, perms]) ** 2).sum(dim=(-1, -2))             # [B, B, N!]
    best = perms[perm_cost.argmin(dim=-1)]                                               # [B, B, N]
    X1 = torch.gather(X1, 2, best.unsqueeze(-1).expand(B, B, N, D))                      # permuted x1

    # batched Kabsch, same formulas as kabsch(): R^T = U diag(1, ..., 1, d) V^T
    U, S, Vh = torch.linalg.svd(X1.transpose(-2, -1) @ X0)                               # [B, B, D, D]
    d = torch.linalg.det(Vh @ U)
    on_diag = torch.ones(B, B, D, device=x0.device, dtype=x0.dtype)
    on_diag[..., -1] = torch.where(d >= 0, 1.0, -1.0)
    X1 = X1 @ (U @ torch.diag_embed(on_diag) @ Vh)                                       # aligned x1, [B, B, N, D]

    cost_matrix = ((X0 - X1) ** 2).sum(dim=(-1, -2))                                     # [B, B]
    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix.cpu().numpy())
    return x0, X1[row_ind, col_ind]
