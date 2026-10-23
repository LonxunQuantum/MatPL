"""CUDA descriptor storage contract and mixed force derivative regression."""
import pytest
import torch
from src.utils.op_loader import load_calc_ops


@pytest.fixture
def ops():
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    return load_calc_ops()


def inputs():
    torch.manual_seed(2915)
    opts = dict(device="cuda", dtype=torch.float64)
    coeff = torch.randn(2, 2, 3, 4, **opts, requires_grad=True)
    rij = torch.tensor([[[1., .6, .8, 0.]], [[1.2, .4, .8, .8]]], **opts)
    nl = torch.tensor([[1], [0]], device="cuda", dtype=torch.int64)
    types = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    return coeff, rij, nl, types


def test_radial_compact_context_and_mixed_derivative(ops):
    coeff, rij, nl, types = inputs()
    feat, dc, dr, noc = ops.calculateNepFeatWithGradContext(
        coeff, rij, nl, types, torch.zeros(2, 3, device="cuda", dtype=torch.float64), 5., 0, 0)
    assert noc.shape == (2, 1, 4, 1)
    seed = torch.randn_like(feat, requires_grad=True)
    force_seed = torch.randn_like(rij)
    vjp = ops.calculateNepFeatInputGrad(seed, coeff, rij, nl, dc, dr, noc, types, 0, 0)
    got = torch.autograd.grad((vjp * force_seed).sum(), coeff, retain_graph=True)[0]
    expected = torch.zeros_like(coeff)
    for i in range(2):
        expected[types[i], types[1-i]] += seed[i, :, None] * noc[i, 0, :, 0] * force_seed[i, 0, 0]
    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-11)
    legacy = torch.zeros(2, 1, 4, 4, device="cuda", dtype=torch.float64)
    legacy[..., 0] = noc[..., 0]
    old_vjp = ops.calculateNepFeatInputGrad(seed, coeff, rij, nl, dc, dr, legacy, types, 0, 0)
    old_grad = torch.autograd.grad((old_vjp * force_seed).sum(), coeff)[0]
    torch.testing.assert_close(got, old_grad, rtol=1e-11, atol=1e-11)


def test_angular_unused_context_is_empty(ops):
    coeff, rij, nl, types = inputs()
    outputs = ops.calculateNepMbFeatWithGradContext(
        coeff, rij, nl, types, torch.zeros(2, 18, device="cuda", dtype=torch.float64),
        0, 4, 2, 1, 5., 0)
    assert all(t.numel() == 0 for t in outputs[1:4])
    seed = torch.randn_like(outputs[0], requires_grad=True)
    vjp = ops.calculateNepMbFeatInputGrad(seed, coeff, rij, nl, *outputs[1:], types,
                                        0, 4, 2, 1, 5., 0)
    grads = torch.autograd.grad(vjp.square().sum(), (seed, coeff))
    assert all(torch.isfinite(g).all() for g in grads)


def test_angular_coefficient_gradient_matches_finite_difference_beyond_one_cta(ops):
    torch.manual_seed(2920)
    opts = dict(device="cuda", dtype=torch.float64)
    coeff = (torch.randn(2, 2, 3, 4, **opts) * 0.03).requires_grad_()
    xyz = torch.randn(4, 70, 3, **opts)
    xyz = xyz / xyz.norm(dim=-1, keepdim=True) * 1.4
    rij = torch.cat([xyz.norm(dim=-1, keepdim=True), xyz], dim=-1)
    nl = torch.arange(280, device="cuda").reshape(4, 70) % 4
    nl[:, -1] = -1
    types = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int64)
    seed = torch.randn(4, 18, **opts)

    def objective(c):
        feat = ops.calculateNepMbFeatWithGradContext(
            c, rij, nl, types, torch.zeros_like(seed), 0, 4, 2, 1, 5., 0)[0]
        return (feat * seed).sum()

    gradient = torch.autograd.grad(objective(coeff), coeff)[0]
    epsilon = 1e-6
    for index in [(0, 0, 0, 0), (0, 1, 2, 3), (1, 0, 1, 2), (1, 1, 0, 0)]:
        plus, minus = coeff.detach().clone(), coeff.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2 * epsilon)
        torch.testing.assert_close(gradient[index], numerical, rtol=2e-7, atol=2e-8)
