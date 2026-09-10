"""Behavior and numerical tests for the required CUDA NEP fitting JIT runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PROBE = r"""
import json
from pathlib import Path
import torch
from src.utils.op_loader import load_calc_ops

ops = load_calc_ops()
reference = torch.empty(1, dtype=torch.float64, device="cuda")
prepared = ops.nep_fitting_jit_prepare(reference, 35, 60, 1)
cache = Path(__import__("os").environ["MATPL_NEP_JIT_CACHE"])
print(json.dumps({
    "prepared": bool(prepared),
    "files": sorted(path.name for path in cache.glob("*")) if cache.exists() else [],
}))
"""


def probe_process(cache_dir: Path):
    env = os.environ.copy()
    env["MATPL_NEP_JIT_CACHE"] = str(cache_dir)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO, env=env,
        text=True, capture_output=True,
    )


def run_probe(cache_dir: Path):
    result = probe_process(cache_dir)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def start_probe(cache_dir: Path):
    env = os.environ.copy()
    env["MATPL_NEP_JIT_CACHE"] = str(cache_dir)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-c", PROBE], cwd=REPO, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def fitting_reference(x, w, b, v, c, atom_ids, offsets):
    """Independent PyTorch expression for energy heads and dE/dfeature."""
    n = x.shape[0]
    atom_types = torch.empty(n, dtype=torch.long, device=x.device)
    for type_index in range(offsets.numel() - 1):
        first, end = int(offsets[type_index]), int(offsets[type_index + 1])
        positions = atom_ids[first:end]
        atom_types[positions] = type_index
    selected_w = w.index_select(0, atom_types)
    hidden = torch.tanh(
        torch.einsum("nd,ndh->nh", x, selected_w)
        + b.index_select(0, atom_types)
    )
    selected_v = v.index_select(0, atom_types)
    energy = torch.einsum("nh,nhq->qn", hidden, selected_v)
    energy = energy + c.index_select(0, atom_types).transpose(0, 1)
    feature_gradient = torch.einsum(
        "nhq,ndh->qnd",
        (1.0 - hidden.square()).unsqueeze(-1) * selected_v,
        selected_w,
    )
    return energy, feature_gradient


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_jit_is_required_when_mode_env_is_unset(tmp_path):
    result = run_probe(tmp_path)
    assert result["prepared"] is True
    assert len(list(tmp_path.glob("*.cubin"))) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_invalid_cache_path_is_an_error(tmp_path):
    invalid_cache = tmp_path / "cache-file"
    invalid_cache.write_text("not a directory")
    result = probe_process(invalid_cache)
    assert result.returncode != 0
    assert "not a directory" in result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
@pytest.mark.parametrize("charge,expected_q", [(False, 1), (True, 2)])
def test_prepare_model_infers_fitting_dimensions(monkeypatch, charge, expected_q):
    from src.model import nep_fused_fitting as adapter

    calls = []

    class FakeOps:
        @staticmethod
        def nep_fitting_jit_prepare(reference, d, h, q):
            calls.append((reference, d, h, q))
            return True

    monkeypatch.setattr(adapter, "_load_raw_ops", lambda: FakeOps())
    hidden = SimpleNamespace(weight=torch.empty(15, 33, dtype=torch.float64))
    if charge:
        net = SimpleNamespace(
            layers=[hidden],
            energy_head=SimpleNamespace(weight=torch.empty(33, 1)),
            charge_head=SimpleNamespace(weight=torch.empty(33, 1)),
        )
    else:
        output = SimpleNamespace(weight=torch.empty(33, 1, dtype=torch.float64))
        net = SimpleNamespace(layers=[hidden, output])
    model = SimpleNamespace(fitting_net=[net])
    assert adapter.prepare_fitting_jit(model) is False
    assert calls == []

    hidden.weight = hidden.weight.cuda()
    if charge:
        net.energy_head.weight = net.energy_head.weight.cuda()
        net.charge_head.weight = net.charge_head.weight.cuda()
    else:
        output.weight = output.weight.cuda()
    assert adapter.prepare_fitting_jit(model) is True
    reference, d, h, q = calls.pop()
    assert reference.device.type == "cuda"
    assert reference.dtype == torch.float64
    assert (d, h, q) == (15, 33, expected_q)


def test_prepare_model_is_called_between_cuda_move_and_ddp_wrap():
    source = (REPO / "src/PWMLFF/nep_network.py").read_text()
    moved = source.index(").to(self.training_type).to(self.device)")
    prepared = source.index("prepare_fitting_jit(model)")
    wrapped = source.index("nn.parallel.DistributedDataParallel(model")
    assert moved < prepared < wrapped


def test_prepare_model_skips_cpu_without_loading_cuda_ops(monkeypatch):
    from src.model import nep_fused_fitting as adapter

    hidden = SimpleNamespace(weight=torch.empty(15, 33, dtype=torch.float64))
    output = SimpleNamespace(weight=torch.empty(33, 1, dtype=torch.float64))
    model = SimpleNamespace(fitting_net=[SimpleNamespace(layers=[hidden, output])])
    monkeypatch.setattr(
        adapter, "_load_raw_ops",
        lambda: pytest.fail("CPU prewarm must not load the CUDA extension"),
    )
    assert adapter.prepare_fitting_jit(model) is False


def test_aot_fitting_implementation_is_removed():
    assert not (REPO / "src/op/kernel/nep_fitting.cu").exists()
    wrapper = (REPO / "src/op/src/nep_fitting_launcher.cpp").read_text()
    runtime = (REPO / "src/op/src/nep_fitting_jit.cpp").read_text()
    assert "template<" not in wrapper
    assert "fitting_atoms<<<" not in wrapper
    assert "module_for_mode" not in runtime
    assert "fallback" not in runtime.lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_jit_forward_and_feature_gradient_match_reference(tmp_path, monkeypatch):
    from src.utils.op_loader import load_calc_ops

    torch.manual_seed(41)
    ops = load_calc_ops()
    counts = [4, 2, 3]
    n, d, h, q = sum(counts), 35, 60, 1
    x = torch.randn(n, d, dtype=torch.float64, device="cuda")
    w = torch.randn(3, d, h, dtype=torch.float64, device="cuda") * 0.1
    b = torch.randn(3, h, dtype=torch.float64, device="cuda") * 0.1
    v = torch.randn(3, h, q, dtype=torch.float64, device="cuda") * 0.1
    c = torch.randn(3, q, dtype=torch.float64, device="cuda") * 0.1
    atom_ids = torch.tensor([8, 2, 5, 0, 7, 1, 6, 4, 3], device="cuda")
    offsets = torch.tensor([0, 4, 6, 9], device="cuda")

    monkeypatch.setenv("MATPL_NEP_JIT_CACHE", str(tmp_path))
    expected = fitting_reference(x, w, b, v, c, atom_ids, offsets)
    actual = ops.nep_fitting_forward(x, w, b, v, c, atom_ids, offsets, counts)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-12, atol=2e-12)
    assert len(list(tmp_path.glob("*.cubin"))) == 1


@pytest.mark.parametrize(
    "d,h,q,seed_mode",
    [(35, 61, 1, "both"), (31, 33, 2, "y"), (96, 100, 2, "g")],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_jit_backward_matches_reference(tmp_path, monkeypatch, d, h, q, seed_mode):
    from src.utils.op_loader import load_calc_ops

    torch.manual_seed(d * 1000 + h * 10 + q)
    ops = load_calc_ops()
    counts = [5, 2, 4]
    n = sum(counts)
    values = [
        torch.randn(n, d, dtype=torch.float64, device="cuda") * 0.1,
        torch.randn(3, d, h, dtype=torch.float64, device="cuda") * 0.1,
        torch.randn(3, h, dtype=torch.float64, device="cuda") * 0.1,
        torch.randn(3, h, q, dtype=torch.float64, device="cuda") * 0.1,
        torch.randn(3, q, dtype=torch.float64, device="cuda") * 0.1,
    ]
    x, w, b, v, c = [value.detach().requires_grad_(True) for value in values]
    atom_ids = torch.tensor([9, 2, 7, 0, 5, 10, 1, 8, 3, 6, 4], device="cuda")
    offsets = torch.tensor([0, 5, 7, 11], device="cuda")
    grad_y = torch.randn(q, n, dtype=torch.float64, device="cuda")
    grad_g = torch.randn(q, n, d, dtype=torch.float64, device="cuda")
    if seed_mode == "y":
        grad_g.zero_()
    elif seed_mode == "g":
        grad_y.zero_()

    expected_y, expected_g = fitting_reference(x, w, b, v, c, atom_ids, offsets)
    expected = torch.autograd.grad(
        (expected_y * grad_y).sum() + (expected_g * grad_g).sum(),
        (x, w, b, v, c),
    )

    monkeypatch.setenv("MATPL_NEP_JIT_CACHE", str(tmp_path))
    assert ops.nep_fitting_jit_prepare(x, d, h, q) is True
    cubin = next(tmp_path.glob("*.cubin"))
    symbols = subprocess.run(
        ["cuobjdump", "--dump-elf-symbols", str(cubin)],
        check=True, text=True, capture_output=True,
    ).stdout
    for name in (
        "fitting_atoms_backward",
        "fitting_parameter_partials",
        "fitting_parameter_reduce",
    ):
        assert name in symbols
    actual = ops.nep_fitting_backward(
        x, w, b, v, c, atom_ids, offsets, counts, grad_y, grad_g
    )
    torch.cuda.synchronize()

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=3e-11, atol=3e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_persistent_cache_reuses_cubin_without_rewrite(tmp_path):
    assert run_probe(tmp_path)["prepared"] is True
    cubin = next(tmp_path.glob("*.cubin"))
    initial_mtime = cubin.stat().st_mtime_ns
    assert run_probe(tmp_path)["prepared"] is True
    assert cubin.stat().st_mtime_ns == initial_mtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_concurrent_prepare_uses_one_locked_cache_entry(tmp_path):
    processes = [start_probe(tmp_path) for _ in range(4)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr
        assert json.loads(stdout.strip().splitlines()[-1])["prepared"] is True
    assert len(list(tmp_path.glob("*.cubin"))) == 1
    assert len(list(tmp_path.glob("*.lock"))) == 1
    assert list(tmp_path.glob("*.tmp.*")) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_corrupt_cubin_is_recompiled_once(tmp_path):
    assert run_probe(tmp_path)["prepared"] is True
    cubin = next(tmp_path.glob("*.cubin"))
    original_size = cubin.stat().st_size
    cubin.write_bytes(b"corrupt")
    assert run_probe(tmp_path)["prepared"] is True
    assert cubin.stat().st_size == original_size
    assert cubin.read_bytes() != b"corrupt"
