"""DW4 system: data loading and energy, eq. (30) of Klein et al. 2023."""

import numpy as np
import torch

from eqfm.flow_matching import remove_mean


def load_dw4(data_dir: str = "data", n_train: int = 100_000, n_test: int = 100_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns centered x_train [n_train, 4, 2] and x_test [n_test, 4, 2], split by the authors' fixed permutation."""
    data = np.load(f"{data_dir}/dw4_data.npy")   # [1e6, 4, 2]
    idx = np.load(f"{data_dir}/dw4_idx.npy")     # [1e6]

    x_train = torch.from_numpy(data[idx[:n_train]])
    x_test = torch.from_numpy(data[idx[-n_test:]])
    return remove_mean(x_train), remove_mean(x_test)


def dw4_energy(x: torch.Tensor, a=0.0, b=-4.0, c=0.9, d0=4.0, tau=1.0) -> torch.Tensor:
    """x: [B, N, D] -> U(x): [B]."""
    N = x.size(1)
    x_i = x.unsqueeze(2).expand(-1, -1, N, -1)                                  # [B, N, N, D]
    x_j = x.unsqueeze(1).expand(-1, N, -1, -1)                                  # [B, N, N, D]

    x_dif = x_i - x_j                                                           # [B, N, N, D]
    d_sq = (x_dif ** 2).sum(-1)                                                 # [B, N, N]
    d = (d_sq + 1e-9) ** 0.5                                                    # [B, N, N]
    d_ij = d - d0

    a_term = d_ij
    b_term = d_ij ** 2
    c_term = b_term ** 2

    diag = torch.eye(N, N, device=x.device).unsqueeze(0)
    mask = torch.ones((1, N, N), device=x.device) - diag

    result = (a * a_term + b * b_term + c * c_term) * mask
    result_sum = result.sum(dim=(1, 2))

    return result_sum / (2 * tau)
