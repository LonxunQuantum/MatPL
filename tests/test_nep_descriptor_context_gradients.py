"""Descriptor contexts must not create extra coefficient-gradient paths."""
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from src.utils.op_loader import load_calc_ops

REPO = Path(__file__).resolve().parents[1]
WHITELIST = REPO / 'src/op/kernel_hip/utilities/nep3_c3_specialization_whitelist.def'
CASES = [tuple(map(int, row.split(','))) for row in re.findall(
    r'^MATPL_NEP_C3_SPECIALIZATION\(([^)]+)\)', WHITELIST.read_text(), re.M)]
CASES += [(9, 6, 118, 4, 2, 1), (8, 4, 3, 4, 2, 0),
          (17, 13, 118, 4, 2, 1), (9, 5, 118, 4, 2, 1)]


def inputs(spec):
    nbase, nmax, ntypes, l3, l4, l5 = spec
    torch.manual_seed(210906 + ntypes + nmax)
    types = torch.tensor([0, ntypes-1, 0, ntypes//2, 1, ntypes-1])
    neighbors = torch.tensor([[j for j in range(6) if j != i] + [-1]
                              for i in range(6)], dtype=torch.int64)
    xyz = torch.randn(6, 6, 3, dtype=torch.float64)
    xyz = xyz / xyz.norm(dim=-1, keepdim=True) * 1.7
    rij = torch.cat((xyz.norm(dim=-1, keepdim=True), xyz), dim=-1)
    coeff = torch.randn(ntypes, ntypes, nmax, nbase, dtype=torch.float64) * .02
    total = 3 + nmax * (l3 + (l4 > 0) + (l5 > 0))
    seeds = torch.randn(6, total, dtype=torch.float64) * .1
    cartesian_probe = torch.randn_like(xyz) * .1
    probe = torch.cat(((cartesian_probe * xyz / rij[:, :, :1]).sum(-1, keepdim=True),
                       cartesian_probe), dim=-1)
    return coeff, rij, neighbors, types, seeds, probe


def evaluate(spec, values, device, derivatives=True):
    nbase, nmax, ntypes, l3, l4, l5 = spec
    coeff, rij, neighbors, types, seeds, probe = [v.to(device) for v in values]
    coeff = coeff.detach().clone().requires_grad_()
    seed = seeds[:, 3:].detach().requires_grad_()
    if device == 'cpu':
        from src.model.nep_net import NEP
        from src.user.input_param import InputParam
        config = {
            'model_type': 'NEP', 'atom_type': list(range(1, ntypes + 1)),
            'device': 'cpu', 'precision': 'float64', 'recover_train': False,
            'model': {'descriptor': {'cutoff': [5., 5.],
                       'n_max': [2, nmax-1], 'basis_size': [2, nbase-1],
                       'l_max': [l3, l4, l5]},
                      'fitting_net': {'network_size': [30, 1]}},
            'optimizer': {'optimizer': 'ADAM', 'train_energy': True, 'train_force': True},
        }
        model = NEP(InputParam(config, 'TRAIN'), [0.] * ntypes,
                    q_scaler=np.ones(seeds.shape[1]), dtype=torch.float64,
                    device=torch.device('cpu'))
        rij = rij.detach().requires_grad_()
        neighbor_types = torch.full_like(neighbors, -1)
        valid = neighbors >= 0
        neighbor_types[valid] = types[neighbors[valid]]
        selected = model.get_c(coeff, nmax-1, nbase-1, types, neighbor_types)
        feature = model.cal_feat_multi_body(rij[:, :, 0], rij[:, :, 1:], types,
                                            selected, nmax-1, nbase-1, 5., .2, l3)
        features = feature
        vjp = torch.autograd.grad(feature, rij, seed, create_graph=True)[0]
    else:
        ops = load_calc_ops(device=device)
        context = ops.calculateNepMbFeatWithGradContext(
            coeff, rij, neighbors, types, torch.zeros_like(seed), 3, l3, l4, l5, 5., 0)
        features = context[0]
        assert features.requires_grad
        assert all(not t.requires_grad for t in context[1:])
        vjp = ops.calculateNepMbFeatInputGrad(
            seed, coeff, rij, neighbors, *context[1:], types, 3, l3, l4, l5, 5., 0)
    objective = (vjp * probe).sum()
    if not derivatives:
        return objective.detach().cpu()
    # The native interface expects a view into the combined radial/angular
    # gradient, as produced by the model's cat backward. Preserve its prefix
    # and row stride instead of creating a compact gradient via multiplication.
    grad_first = torch.autograd.grad(features, coeff, seed,
                                     retain_graph=True)[0]
    grad_seed, grad_coeff = torch.autograd.grad(objective, (seed, coeff))
    # Compare Cartesian derivatives: the CPU and native formulas can differ
    # away from the physical r=norm(xyz) constraint in their four partials.
    cartesian_vjp = vjp[:, :, 1:] + vjp[:, :, :1] * rij[:, :, 1:] / rij[:, :, :1]
    return tuple(v.detach().cpu() for v in (features, cartesian_vjp, grad_first,
                                           grad_seed, grad_coeff))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires an allocated GPU')
@pytest.mark.parametrize('spec', CASES)
def test_descriptor_context_mixed_derivatives_match_cpu(spec):
    values = inputs(spec)
    expected = evaluate(spec, values, 'cpu')
    actual = evaluate(spec, values, 'cuda')
    for name, got, want in zip(('features', 'vjp', 'first_coeff',
                               'second_seed', 'second_coeff'), actual, expected):
        # The existing grad-seed kernels accumulate in float; preserve their
        # documented numerical boundary while checking coefficient paths tightly.
        if name == 'second_seed':
            torch.testing.assert_close(got, want, rtol=3e-6, atol=3e-7)
        else:
            torch.testing.assert_close(got, want, rtol=2e-8, atol=2e-9)
    nbase, nmax, ntypes, *_ = spec
    for index in ((0, ntypes-1, nmax-1, nbase-1), (ntypes-1, 0, 0, 0)):
        plus, minus = values[0].clone(), values[0].clone()
        epsilon = 1e-6
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (evaluate(spec, (plus, *values[1:]), 'cuda', False)
                     - evaluate(spec, (minus, *values[1:]), 'cuda', False)) / (2 * epsilon)
        torch.testing.assert_close(actual[-1][index], numerical, rtol=3e-6, atol=2e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires an allocated GPU')
def test_radial_context_outputs_are_nondifferentiable():
    coeff, rij, neighbors, types, seeds, _ = [v.cuda() for v in inputs(CASES[0])]
    coeff.requires_grad_()
    ops = load_calc_ops(device='cuda')
    context = ops.calculateNepFeatWithGradContext(
        coeff, rij, neighbors, types, torch.zeros(6, 13, device='cuda', dtype=torch.float64),
        5., 3, 0)
    assert context[0].requires_grad
    assert all(not t.requires_grad for t in context[1:])
    full_seed = torch.randn(6, 16, device='cuda', dtype=torch.float64)
    seed = full_seed[:, :13].detach().requires_grad_()
    vjp = ops.calculateNepFeatInputGrad(
        seed, coeff, rij, neighbors, *context[1:], types, 3, 0)
    force_seed = torch.randn_like(rij)
    actual = torch.autograd.grad((vjp * force_seed).sum(), coeff)[0]
    expected = torch.zeros_like(coeff)
    for i in range(6):
        for j in range(5):
            neighbor_type = types[neighbors[i, j]]
            expected[types[i], neighbor_type] += (
                seed[i, :, None] * context[-1][i, j, :, 0]
                * force_seed[i, j, 0])
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
