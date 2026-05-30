"""
Lightweight NVTX range helpers for MatPL profiling.

Usage:
    from src.utils.nvtx_helper import nvtx_range

    with nvtx_range("forward"):
        ...

NVTX ranges have near-zero overhead (~ns) when no profiler is attached.
Disable entirely by setting env MATPL_NVTX_ENABLED=0.
"""
import os
import torch
from contextlib import contextmanager

_NVTX_ENABLED = os.environ.get("MATPL_NVTX_ENABLED", "1") != "0"
_HAS_CUDA = torch.cuda.is_available()


@contextmanager
def nvtx_range(name: str):
    if _NVTX_ENABLED and _HAS_CUDA:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if _NVTX_ENABLED and _HAS_CUDA:
            torch.cuda.nvtx.range_pop()
