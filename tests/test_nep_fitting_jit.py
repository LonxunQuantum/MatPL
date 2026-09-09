"""Behavior tests for the CUDA NEP fitting JIT runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys

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


def run_probe(cache_dir: Path, mode: str):
    env = os.environ.copy()
    env["MATPL_NEP_FITTING_JIT"] = mode
    env["MATPL_NEP_JIT_CACHE"] = str(cache_dir)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO, env=env,
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def start_probe(cache_dir: Path, mode: str):
    env = os.environ.copy()
    env["MATPL_NEP_FITTING_JIT"] = mode
    env["MATPL_NEP_JIT_CACHE"] = str(cache_dir)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-c", PROBE], cwd=REPO, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_disabled_mode_does_not_touch_cache(tmp_path):
    result = run_probe(tmp_path, mode="0")
    assert result["prepared"] is False
    assert result["files"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_jit_forward_and_feature_gradient_match_aot(tmp_path, monkeypatch):
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
    monkeypatch.setenv("MATPL_NEP_FITTING_JIT", "0")
    expected = ops.nep_fitting_forward(x, w, b, v, c, atom_ids, offsets, counts)
    torch.cuda.synchronize()

    monkeypatch.setenv("MATPL_NEP_FITTING_JIT", "1")
    assert ops.nep_fitting_jit_prepare(x, d, h, q) is True
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
def test_jit_backward_matches_aot(tmp_path, monkeypatch, d, h, q, seed_mode):
    from src.utils.op_loader import load_calc_ops

    torch.manual_seed(d * 1000 + h * 10 + q)
    ops = load_calc_ops()
    counts = [5, 2, 4]
    n = sum(counts)
    x = torch.randn(n, d, dtype=torch.float64, device="cuda") * 0.1
    w = torch.randn(3, d, h, dtype=torch.float64, device="cuda") * 0.1
    b = torch.randn(3, h, dtype=torch.float64, device="cuda") * 0.1
    v = torch.randn(3, h, q, dtype=torch.float64, device="cuda") * 0.1
    c = torch.randn(3, q, dtype=torch.float64, device="cuda") * 0.1
    atom_ids = torch.tensor([9, 2, 7, 0, 5, 10, 1, 8, 3, 6, 4], device="cuda")
    offsets = torch.tensor([0, 5, 7, 11], device="cuda")
    grad_y = torch.randn(q, n, dtype=torch.float64, device="cuda")
    grad_g = torch.randn(q, n, d, dtype=torch.float64, device="cuda")
    if seed_mode == "y":
        grad_g.zero_()
    elif seed_mode == "g":
        grad_y.zero_()

    monkeypatch.setenv("MATPL_NEP_JIT_CACHE", str(tmp_path))
    monkeypatch.setenv("MATPL_NEP_FITTING_JIT", "0")
    expected = ops.nep_fitting_backward(
        x, w, b, v, c, atom_ids, offsets, counts, grad_y, grad_g
    )
    torch.cuda.synchronize()

    monkeypatch.setenv("MATPL_NEP_FITTING_JIT", "1")
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
    assert run_probe(tmp_path, "1")["prepared"] is True
    cubin = next(tmp_path.glob("*.cubin"))
    initial_mtime = cubin.stat().st_mtime_ns
    assert run_probe(tmp_path, "1")["prepared"] is True
    assert cubin.stat().st_mtime_ns == initial_mtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_concurrent_prepare_uses_one_locked_cache_entry(tmp_path):
    processes = [start_probe(tmp_path, "1") for _ in range(4)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr
        assert json.loads(stdout.strip().splitlines()[-1])["prepared"] is True
    assert len(list(tmp_path.glob("*.cubin"))) == 1
    assert len(list(tmp_path.glob("*.lock"))) == 1
    assert list(tmp_path.glob("*.tmp.*")) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_corrupt_cubin_is_recompiled_once(tmp_path):
    assert run_probe(tmp_path, "1")["prepared"] is True
    cubin = next(tmp_path.glob("*.cubin"))
    original_size = cubin.stat().st_size
    cubin.write_bytes(b"corrupt")
    assert run_probe(tmp_path, "1")["prepared"] is True
    assert cubin.stat().st_size == original_size
    assert cubin.read_bytes() != b"corrupt"
