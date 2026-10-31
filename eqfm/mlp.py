"""Non-equivariant baseline: an MLP on flattened coordinates (no S(N) or O(D) symmetry built in)."""

import torch
import torch.nn as nn

from eqfm.egnn import MLPSequence


class MLPVectorField(nn.Module):
    """v_theta(t, x): [B], [B, N, D] -> [B, N, D]. Sees the configuration as a flat vector of length N*D plus t.

    The output is projected to zero center of mass, so samples stay in the same subspace as for the EGNN;
    this keeps translation handling identical and leaves rotations and permutations as the only difference.
    """

    def __init__(self, n_particles: int, dim: int, hidden_dim: int = 108, n_layers: int = 4):
        super().__init__()
        self.n_particles, self.dim = n_particles, dim
        self.net = MLPSequence(n_particles * dim + 1, hidden_dim, n_particles * dim, n_layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        t = t.reshape(B, 1)
        v = self.net(torch.cat([x.reshape(B, -1), t], dim=-1)).reshape(B, self.n_particles, self.dim)
        return v - v.mean(dim=1, keepdim=True)
