"""Equivariance tests for EGNNVectorField. Run: pytest test_egnn.py"""

import pytest
import torch

from eqfm.egnn import EGNNVectorField


def random_orthogonal(D, reflection=False):
    Q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
    if (torch.det(Q) < 0) != reflection:   # det = +1: rotation, det = -1: reflection
        Q[:, 0] = -Q[:, 0]
    return Q


def check_vector_field(N, D, B=8, H=32, L=3, atol=1e-10):
    torch.manual_seed(0)
    model = EGNNVectorField(H, L).double()
    t = torch.rand(B, dtype=torch.float64)
    x = torch.randn(B, N, D, dtype=torch.float64)
    x = x - x.mean(dim=1, keepdim=True)
    v = model(t, x)
    assert v.shape == (B, N, D)

    # O(D): rotation and reflection
    for reflection in (False, True):
        R = random_orthogonal(D, reflection)
        assert torch.allclose(model(t, x @ R.T), v @ R.T, atol=atol), f"rotation/reflection={reflection}"

    # S(N): particle permutation
    perm = torch.randperm(N)
    assert torch.allclose(model(t, x[:, perm]), v[:, perm], atol=atol), "permutation"

    # translation: v(t, x + c) = v(t, x)
    c = torch.randn(1, 1, D, dtype=torch.float64)
    assert torch.allclose(model(t, x + c), v, atol=atol), "translation"

    # zero center of mass of the velocity
    assert torch.allclose(v.mean(dim=1), torch.zeros(B, D, dtype=torch.float64), atol=atol), "center of mass"

    # no NaNs in gradients (sqrt on the diagonal)
    x_req = x.clone().requires_grad_(True)
    model(t, x_req).square().sum().backward()
    assert not torch.isnan(x_req.grad).any(), "NaN in grad wrt x"
    assert all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None), "NaN in param grads"



@pytest.mark.parametrize("N", [4, 13])
@pytest.mark.parametrize("D", [2, 3])
def test_vector_field(N, D):
    check_vector_field(N, D)
