"""Tests for the couplings: minibatch OT, Kabsch, alignment and equivariant OT. Run: pytest tests/"""

import pytest
import torch

from eqfm.couplings import align, eq_ot_coupling, eq_ot_coupling_bruteforce, kabsch, ot_coupling
from eqfm.dw4 import dw4_energy
from eqfm.flow_matching import remove_mean, sample_prior


def random_rotation(D, angle_scale=None):
    """Random R in SO(D). angle_scale=None: any rotation; a small number: a rotation close to the identity."""
    if angle_scale is None:
        Q, _ = torch.linalg.qr(torch.randn(D, D))
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q
    A = torch.randn(D, D) * angle_scale
    return torch.matrix_exp(A - A.T)       # exp of a skew-symmetric matrix is a rotation


def random_reflection(D):
    R = random_rotation(D)
    R[:, 0] = -R[:, 0]                     # det = -1
    return R


def mean_cost(x0, x1):
    return ((x0 - x1) ** 2).sum(dim=(1, 2)).mean().item()


@pytest.fixture
def batch():
    """A batch of DW4-like centered configurations (structured, not pure noise) and prior samples."""
    torch.manual_seed(0)
    B, N, D = 64, 4, 2
    square = torch.tensor([[2., 2.], [2., -2.], [-2., 2.], [-2., -2.]])
    x1 = remove_mean(square @ torch.stack([random_rotation(D) for _ in range(B)]) + 0.3 * torch.randn(B, N, D))
    return sample_prior(B, N, D), x1


def test_ot_is_a_reordering(batch):
    x0, x1 = batch
    x0_ot, x1_ot = ot_coupling(x0, x1)
    assert torch.equal(x0_ot, x0)
    assert torch.equal(x1_ot.flatten(1).sort(dim=0).values, x1.flatten(1).sort(dim=0).values)


@pytest.mark.parametrize("N, D", [(4, 2), (13, 3)])
def test_kabsch_recovers_rotation_not_reflection(N, D):
    torch.manual_seed(0)
    x = torch.randn(N, D)
    x = x - x.mean(dim=0)                   # one configuration [N, D]: center over particles
    for _ in range(10):
        R = random_rotation(D)
        assert torch.allclose(kabsch(x, x @ R.T), x, atol=1e-5)
    assert (kabsch(x, x @ random_reflection(D).T) - x).abs().max() > 1e-2      # R must be in SO(D)


@pytest.mark.parametrize("N, D", [(4, 2), (13, 3)])
def test_align_recovers_permutation_and_small_rotation(N, D):
    torch.manual_seed(0)
    x = torch.randn(N, D)
    x = x - x.mean(dim=0)                   # one configuration [N, D]: center over particles
    for _ in range(10):
        perm = torch.randperm(N)
        assert torch.allclose(align(x, x[perm]), x, atol=1e-5)
        R = random_rotation(D, angle_scale=0.05)
        assert torch.allclose(align(x, (x @ R.T)[perm]), x, atol=1e-5)


def test_eq_ot_keeps_samples_valid(batch):
    x0, x1 = batch
    x0_eq, x1_eq = eq_ot_coupling(x0, x1)
    assert torch.equal(x0_eq, x0)
    assert torch.allclose(dw4_energy(x1_eq).sort().values, dw4_energy(x1).sort().values, atol=1e-3)
    assert x1_eq.mean(dim=1).abs().max() < 1e-5


def test_transport_cost_decreases(batch):
    x0, x1 = batch
    independent = mean_cost(x0, x1)
    ot = mean_cost(*ot_coupling(x0, x1))
    eq_ot = mean_cost(*eq_ot_coupling(x0, x1))
    assert independent > ot > eq_ot


def test_bruteforce_matches_loop(batch):
    x0, x1 = batch
    _, a = eq_ot_coupling(x0, x1)
    _, b = eq_ot_coupling_bruteforce(x0, x1)
    assert abs(mean_cost(x0, a) - mean_cost(x0, b)) < 1e-4
    assert torch.allclose(a, b, atol=1e-4)
