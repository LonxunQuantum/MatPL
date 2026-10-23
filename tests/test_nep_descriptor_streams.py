"""Descriptor double backward must return both gradients on the caller's stream."""
import pytest
import torch

from src.utils.op_loader import load_calc_ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an allocated GPU")
@pytest.mark.parametrize("angular", [False, True])
def test_double_backward_gradients_follow_nondefault_stream(angular):
    ops = load_calc_ops()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(2196)
        options = dict(device="cuda", dtype=torch.float64)
        # Six angular orders exercise the HIP recomputation specialization.
        coeff = (torch.randn(2, 2, 6, 9, **options) * 0.03).requires_grad_()
        xyz = torch.tensor([[[1.2, .3, .2]], [[-1.2, -.3, -.2]]], **options)
        rij = torch.cat((xyz.norm(dim=-1, keepdim=True), xyz), dim=-1)
        neighbors = torch.tensor([[1], [0]], device="cuda", dtype=torch.int64)
        types = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        count = 36 if angular else 6
        if angular:
            context = ops.calculateNepMbFeatWithGradContext(
                coeff, rij, neighbors, types, torch.zeros(2, count, **options),
                0, 4, 2, 1, 5., 0)
        else:
            context = ops.calculateNepFeatWithGradContext(
                coeff, rij, neighbors, types, torch.zeros(2, count, **options),
                5., 0, 0)
        seed = torch.randn_like(context[0], requires_grad=True)
        if angular:
            vjp = ops.calculateNepMbFeatInputGrad(
                seed, coeff, rij, neighbors, *context[1:], types,
                0, 4, 2, 1, 5., 0)
        else:
            vjp = ops.calculateNepFeatInputGrad(
                seed, coeff, rij, neighbors, *context[1:], types, 0, 0)
        cotangent = torch.randn_like(vjp)
        # Setup still includes legacy HIP producers on the default stream.
        torch.cuda.synchronize()
        expected = torch.autograd.grad(vjp, (seed, coeff), cotangent,
                                       retain_graph=True)
    torch.cuda.synchronize()
    # Warm allocations above so allocator synchronization cannot hide the race.
    with torch.cuda.stream(torch.cuda.default_stream()):
        torch.cuda._sleep(100_000_000)
    with torch.cuda.stream(stream):
        actual = torch.autograd.grad(vjp, (seed, coeff), cotangent)
        observed = tuple(value.clone() for value in actual)
    torch.cuda.synchronize()
    for got, want in zip(observed, expected):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-9)
