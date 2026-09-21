"""The per-image energy reduction must preserve first and second derivatives."""
import pytest
import torch

from src.model.nep_net import _SegmentSum


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an allocated GPU")
    return request.param


@pytest.mark.parametrize("sizes", [[3, 1, 5], [1], [0, 2, 0, 3]])
def test_energy_segment_sum_matches_split_values_and_gradients(device, sizes):
    torch.manual_seed(2109)
    values = torch.randn(sum(sizes), dtype=torch.float64, device=device,
                         requires_grad=True)
    lengths = torch.tensor(sizes, device=device, dtype=torch.int64)
    actual = _SegmentSum.apply(values, lengths)
    expected = torch.stack([part.sum() for part in values.split(sizes)])
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual.square().sum(), values, create_graph=True)[0]
    want = torch.autograd.grad(expected.square().sum(), values, create_graph=True)[0]
    torch.testing.assert_close(got, want)
    direction = torch.randn_like(values)
    got_second = torch.autograd.grad((got * direction).sum(), values)[0]
    want_second = torch.autograd.grad((want * direction).sum(), values)[0]
    torch.testing.assert_close(got_second, want_second)


def test_energy_segment_sum_passes_gradcheck_and_gradgradcheck(device):
    values = torch.randn(6, device=device, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([2, 1, 3], device=device, dtype=torch.int64)
    function = lambda x: _SegmentSum.apply(x, lengths)
    assert torch.autograd.gradcheck(function, (values,))
    assert torch.autograd.gradgradcheck(function, (values,))
