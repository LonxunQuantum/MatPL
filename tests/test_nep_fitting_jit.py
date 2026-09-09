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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Slurm GPU allocation")
def test_disabled_mode_does_not_touch_cache(tmp_path):
    result = run_probe(tmp_path, mode="0")
    assert result["prepared"] is False
    assert result["files"] == []
