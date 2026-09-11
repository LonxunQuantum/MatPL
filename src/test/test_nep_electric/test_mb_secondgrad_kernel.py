import dataclasses
import os
import re

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


def _single_type_case(n_max=2, n_base=2, lmax=(1, 0, 0)):
    return CaseSpec((0, 0), ((1,), (0,)), n_max, n_base, 0, *lmax)


def _repeated_type_case(n_types=3, n_max=5, n_base=9, lmax=(4, 2, 1)):
    types = tuple(range(n_types)) * 2
    neighbors = tuple(tuple(j for j in range(len(types)) if j != i) for i in range(len(types)))
    return CaseSpec(types, neighbors, n_max, n_base, 0, *lmax)


def _six_local_type_case(n_max=5, n_base=9, lmax=(4, 2, 1)):
    types = (0, 0, 1, 2, 3, 4, 5)
    neighbors = tuple(tuple(j for j in range(7) if j != i) for i in range(7))
    return CaseSpec(types, neighbors, n_max, n_base, 0, *lmax)


def _wide_neighbor_case(valid_neighbors=39, max_neighbors=43, n_max=5, n_base=9, lmax=(4, 2, 1)):
    natoms = valid_neighbors + 1
    types = tuple(i % 3 for i in range(natoms))
    neighbors = tuple(
        tuple(j for j in range(natoms) if j != i) + (-1,) * (max_neighbors - valid_neighbors)
        for i in range(natoms)
    )
    return CaseSpec(types, neighbors, n_max, n_base, 0, *lmax)


def _empty_slot_case(max_neighbors=8, n_max=5, n_base=9, lmax=(4, 2, 1)):
    return CaseSpec(
        (0, 1, 2),
        ((1,) + (-1,) * (max_neighbors - 1),
         (0,) + (-1,) * (max_neighbors - 1),
         (-1,) * max_neighbors),
        n_max, n_base, 0, *lmax,
    )


def _make_case(case):
    torch.manual_seed(20260908)
    device, dtype = torch.device("cuda"), torch.float64
    atom_map = torch.tensor(case.atom_types, dtype=torch.int64, device=device)
    nl = torch.tensor(case.neighbors, dtype=torch.int64, device=device)
    natoms, max_neighbors = nl.shape
    coords = torch.randn(natoms, max_neighbors, 3, dtype=dtype, device=device) * 0.25
    distance = 0.8 + 2.7 * torch.rand(natoms, max_neighbors, 1, dtype=dtype, device=device)
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
    # The custom VJP takes the many-body view of a full descriptor seed. Keep
    # the full row stride and storage offset expected by the CUDA launchers;
    # its backward returns only the many-body columns.
    seed_storage = torch.randn_like(feats)
    seed = seed_storage[:, case.feat_2b_num:].detach().requires_grad_(True)
    probe = torch.randn_like(d12)
    return coeff, d12, nl, atom_map, feats, seed, probe


def _coefficient_second_grad(case, mode, *, delay_default_stream=False):
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
        if delay_default_stream:
            # Finish setup before introducing the stream-ordering challenge.
            # The custom VJP node was created on the caller's non-default
            # stream, which is also where autograd must run its backward.
            torch.cuda.synchronize()
            with torch.cuda.stream(torch.cuda.default_stream()):
                torch.cuda._sleep(100_000_000)
        grad_seed, grad_coeff = torch.autograd.grad(loss, (seed, coeff))
        # Consume both outputs on the calling stream before any device-wide
        # synchronization can conceal a kernel launched on the wrong stream.
        result = loss.detach(), grad_seed.detach().clone(), grad_coeff.detach().clone()
        torch.cuda.synchronize()
        return result
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


def test_optimized_secondgrad_mode_reports_unsupported_specialization():
    with pytest.raises(RuntimeError, match="unsupported optimized NEP"):
        _coefficient_second_grad(_single_type_case(), "optimized")


@pytest.mark.parametrize("lmax", [(4, 0, 0), (4, 2, 0), (4, 2, 1)])
def test_auto_matches_legacy_for_body_combinations(lmax):
    case = _repeated_type_case(n_max=4, n_base=8, lmax=lmax)
    _assert_triplet_close(
        _coefficient_second_grad(case, "auto"),
        _coefficient_second_grad(case, "legacy"),
    )


@pytest.mark.parametrize("n_max,n_base,lmax", [
    (4, 9, (4, 2, 1)),
    (5, 8, (4, 2, 1)),
    (5, 9, (3, 2, 1)),
    (5, 9, (4, 0, 1)),
    (5, 9, (4, 2, 0)),
], ids=["radial", "basis", "three-body", "four-body", "five-body"])
def test_forced_optimized_rejects_unsupported_shape(n_max, n_base, lmax):
    case = _single_type_case(n_max=n_max, n_base=n_base, lmax=lmax)
    with pytest.raises(RuntimeError, match="unsupported optimized NEP") as error:
        _coefficient_second_grad(case, "optimized")
    message = str(error.value)
    for key, value in zip(
        ("n_max_3b", "n_base_3b", "lmax_3", "lmax_4", "lmax_5"),
        (n_max, n_base, *lmax),
    ):
        assert f"{key}={value}" in message
    required = re.search(r"required_shared_bytes=(\d+)", message)
    available = re.search(r"available_shared_bytes=(\d+)", message)
    assert required and int(required.group(1)) > 0
    assert available and int(available.group(1)) > 0


def test_auto_matches_optimized_for_omat24_shape():
    case = _six_local_type_case()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as profiler:
        automatic = _coefficient_second_grad(case, "auto")
    kernel_names = [event.key for event in profiler.key_averages()]
    assert any("nep_mb_secondgrad_fused" in name for name in kernel_names), kernel_names
    _assert_triplet_close(
        automatic,
        _coefficient_second_grad(case, "optimized"),
    )


@pytest.mark.parametrize("case", [
    _single_type_case(n_max=5, n_base=9, lmax=(4, 2, 1)),
    _repeated_type_case(),
    _six_local_type_case(),
    _wide_neighbor_case(),
    _wide_neighbor_case(valid_neighbors=65, max_neighbors=67),
    _empty_slot_case(),
    dataclasses.replace(_repeated_type_case(), feat_2b_num=6),
], ids=[
    "single-type", "repeated-types", "six-local-types",
    "wide-neighbor-43", "wide-neighbor-67",
    "empty-slots", "radial-prefix",
])
def test_optimized_matches_legacy(case):
    legacy = _coefficient_second_grad(case, "legacy")
    optimized = _coefficient_second_grad(case, "optimized")
    _assert_triplet_close(optimized, legacy)


@pytest.mark.parametrize("case", [
    dataclasses.replace(_repeated_type_case(), feat_2b_num=6),
    _wide_neighbor_case(),
    _wide_neighbor_case(valid_neighbors=65, max_neighbors=67),
], ids=[
    "radial-prefix", "wide-neighbor-43", "wide-neighbor-67",
])
def test_optimized_secondgrad_obeys_current_stream(case):
    legacy = _coefficient_second_grad(case, "legacy")
    stream = torch.cuda.Stream()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as profiler:
        with torch.cuda.stream(stream):
            optimized = _coefficient_second_grad(case, "optimized", delay_default_stream=True)
    kernel_names = [event.key for event in profiler.key_averages()]
    cta_pattern = r"nep_mb_secondgrad_fused.*(?:\(int\)64|, 64>)"
    assert any(re.search(cta_pattern, name) for name in kernel_names), kernel_names
    _assert_triplet_close(optimized, legacy)
