"""NEP regularization must retain the historical parameter set and gradients."""
import pytest
import torch

from src.loss import loss


def _model(dtype=torch.float64):
    # Large sentinels make accidental inclusion of the first two params visible.
    return torch.nn.ParameterList([
        torch.nn.Parameter(torch.tensor([100.0], dtype=dtype)),
        torch.nn.Parameter(torch.tensor([-200.0], dtype=dtype)),
        torch.nn.Parameter(torch.tensor([[-2.0, 0.0], [1.0, 3.0]], dtype=dtype).t()),
        torch.nn.Parameter(torch.tensor([-4.0], dtype=dtype)),
    ])


def test_nep_metrics_preserve_values_parameter_set_and_gradients():
    assert hasattr(loss, "nep_l1_l2"), "NEP needs a batched regularization helper"
    model = _model()
    l1, l2 = loss.nep_l1_l2(model)
    assert l1.dtype == l2.dtype == torch.float64
    torch.testing.assert_close(l1, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(l2, torch.tensor(6.0, dtype=torch.float64))
    (l1 + l2).backward()
    assert model[0].grad is None and model[1].grad is None
    torch.testing.assert_close(model[2].grad, torch.tensor([[-1.0, 0.6], [0.0, 1.4]], dtype=torch.float64))
    torch.testing.assert_close(model[3].grad, torch.tensor([-1.8], dtype=torch.float64))


def test_nep_metrics_are_recomputed_after_parameter_update():
    model = _model()
    loss.nep_l1_l2(model)
    with torch.no_grad():
        model[3].fill_(1.0)
    l1, l2 = loss.nep_l1_l2(model)
    torch.testing.assert_close(l1, torch.tensor(1.4, dtype=torch.float64))
    torch.testing.assert_close(l2, torch.tensor(3.0, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nep_metrics_match_legacy_for_many_parameters(dtype):
    torch.manual_seed(48)
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn(13, dtype=dtype)) for _ in range(257)
    ])
    expected = loss.print_l1_l2(model)
    actual = loss.nep_l1_l2(model)
    for old, new in zip(expected, actual):
        torch.testing.assert_close(new, old)
    old_grads = torch.autograd.grad(sum(expected), tuple(model), allow_unused=True)
    new_grads = torch.autograd.grad(sum(actual), tuple(model), allow_unused=True)
    for old, new in zip(old_grads, new_grads):
        if old is None:
            assert new is None
        else:
            torch.testing.assert_close(new, old)


def test_nep_metrics_keep_legacy_empty_and_mixed_dtype_behavior():
    for model in [
        torch.nn.ParameterList(list(_model())[:2]),
        torch.nn.ParameterList(list(_model())[:2] + [torch.nn.Parameter(torch.ones(2, dtype=torch.float32))]),
    ]:
        for old, new in zip(loss.print_l1_l2(model), loss.nep_l1_l2(model)):
            torch.testing.assert_close(new, old, equal_nan=True)
