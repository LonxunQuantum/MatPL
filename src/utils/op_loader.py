"""Load the CalcOps library built for the active PyTorch backend."""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch


_LOADED_NAMESPACES: Dict[Tuple[int, str], Any] = {}
_SUPPORTED_BACKENDS = {"cpu", "cuda", "hip"}


def detect_compiled_backend(torch_module=torch) -> str:
    """Return the accelerator backend supported by this PyTorch build."""
    version = torch_module.version
    if getattr(version, "hip", None):
        return "hip"
    if getattr(version, "cuda", None):
        return "cuda"
    return "cpu"


def select_runtime_backend(torch_module=torch) -> str:
    """Choose GPU when it is compiled and available, otherwise choose CPU."""
    compiled_backend = detect_compiled_backend(torch_module)
    if compiled_backend != "cpu" and torch_module.cuda.is_available():
        return compiled_backend
    return "cpu"


def get_library_path(backend: str, src_root: Path | None = None) -> Path:
    """Return the isolated build-tree path for a CalcOps backend library."""
    normalized_backend = backend.lower()
    if normalized_backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported CalcOps backend {backend!r}; expected cpu, cuda, or hip"
        )

    root = Path(src_root) if src_root is not None else Path(__file__).resolve().parents[1]
    library_name = (
        "libCalcOps_bind_cpu.so"
        if normalized_backend == "cpu"
        else "libCalcOps_bind.so"
    )
    return root / "op" / "build" / normalized_backend / "lib" / library_name


def load_calc_ops(torch_module=torch, src_root: Path | None = None):
    """Load CalcOps once and return its CPU- or GPU-compatible namespace."""
    backend = select_runtime_backend(torch_module)
    library_path = get_library_path(backend, src_root)
    cache_key = (id(torch_module), str(library_path))
    if cache_key in _LOADED_NAMESPACES:
        return _LOADED_NAMESPACES[cache_key]

    if not library_path.is_file():
        raise FileNotFoundError(
            f"CalcOps {backend} library not found at {library_path}. "
            f"Build it with ./src/build.sh --gpu-backend {backend}."
        )

    torch_module.ops.load_library(str(library_path))
    namespace_name = "CalcOps_cpu" if backend == "cpu" else "CalcOps_cuda"
    try:
        namespace = getattr(torch_module.ops, namespace_name)
    except AttributeError as error:
        raise RuntimeError(
            f"{library_path} did not register torch.ops.{namespace_name}"
        ) from error

    _LOADED_NAMESPACES[cache_key] = namespace
    return namespace
