import dataclasses
import os

import pytest
import torch

from src.utils.op_loader import load_calc_ops

try:
    CalcOps = load_calc_ops()
except FileNotFoundError:
    CalcOps = None
except OSError as exc:
    if "libcuda.so" not in str(exc):
        raise
    CalcOps = None


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    atom_types: tuple[int, ...]
    neighbors: tuple[tuple[int, ...], ...]
    n_max: int
    n_base: int
    feat_2b_num: int
    lmax_3: int
    lmax_4: int
    lmax_5: int


def _single_type_case():
    return CaseSpec((0, 0), ((1,), (0,)), 2, 2, 0, 1, 0, 0)


def _make_case(case):
    torch.manual_seed(20260908)
    device, dtype = torch.device("cuda"), torch.float64
    atom_map = torch.tensor(case.atom_types, dtype=torch.int64, device=device)
    nl = torch.tensor(case.neighbors, dtype=torch.int64, device=device)
    natoms, max_neighbors = nl.shape
    coords = torch.randn(natoms, max_neighbors, 3, dtype=dtype, device=device) * 0.25
    distance = coords.square().sum(-1, keepdim=True).sqrt() + 0.8
    d12 = torch.cat((distance, coords), dim=-1).detach().requires_grad_(True)
    ntypes = max(case.atom_types) + 1
    coeff = torch.randn(
        ntypes, ntypes, case.n_max, case.n_base,
        dtype=dtype, device=device, requires_grad=True,
    )
    many_body = case.n_max * (
        case.lmax_3 + int(case.lmax_4 > 0) + int(case.lmax_5 > 0)
    )
    feats = torch.zeros(
        natoms, case.feat_2b_num + many_body, dtype=dtype, device=device,
        requires_grad=True,
    )
    seed = torch.randn_like(feats, requires_grad=True)
    probe = torch.randn_like(d12)
    return coeff, d12, nl, atom_map, feats, seed, probe


def _coefficient_second_grad(case, mode):
    if CalcOps is None or not torch.cuda.is_available():
        pytest.skip("CUDA extension is unavailable")
    previous = os.environ.get("MATPL_NEP_MB_SECONDGRAD_MODE")
    os.environ["MATPL_NEP_MB_SECONDGRAD_MODE"] = mode
    try:
        coeff, d12, nl, atom_map, feats, seed, probe = _make_case(case)
        feat, dc3, d3, d3_noc, sums = CalcOps.calculateNepMbFeatWithGradContext(
            coeff, d12, nl, atom_map, feats,
            case.feat_2b_num, case.lmax_3, case.lmax_4, case.lmax_5,
            5.0, 0,
        )
        vjp = CalcOps.calculateNepMbFeatInputGrad(
            seed, coeff, d12, nl, dc3, d3, d3_noc, sums, atom_map,
            case.feat_2b_num, case.lmax_3, case.lmax_4, case.lmax_5,
            5.0, 0,
        )
        loss = (vjp * probe).sum()
        grad_seed, grad_coeff = torch.autograd.grad(loss, (seed, coeff))
        torch.cuda.synchronize()
        return loss.detach(), grad_seed.detach(), grad_coeff.detach()
    finally:
        if previous is None:
            os.environ.pop("MATPL_NEP_MB_SECONDGRAD_MODE", None)
        else:
            os.environ["MATPL_NEP_MB_SECONDGRAD_MODE"] = previous


def _assert_triplet_close(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected):
        absolute = (actual_tensor - expected_tensor).abs()
        relative = absolute / expected_tensor.abs().clamp_min(1e-30)
        print(f"max_abs={absolute.max().item():.6e} max_rel={relative.max().item():.6e}")
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-9, atol=1e-11)


def test_invalid_secondgrad_mode_is_rejected():
    with pytest.raises(RuntimeError, match="MATPL_NEP_MB_SECONDGRAD_MODE"):
        _coefficient_second_grad(_single_type_case(), "invalid")


def test_optimized_secondgrad_mode_reports_unavailable_specialization():
    with pytest.raises(RuntimeError, match="specialization is unavailable"):
        _coefficient_second_grad(_single_type_case(), "optimized")
