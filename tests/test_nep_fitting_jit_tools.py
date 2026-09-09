import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    path = REPO / "tests" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resource_parser_reports_all_jit_kernels_and_spill_storage():
    checker = load_script("check_nep_fitting_jit_cubin.py")
    text = """
 Function fitting_parameter_reduce:
  REG:39 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:400
 Function fitting_parameter_partials:
  REG:168 STACK:0 SHARED:27712 LOCAL:0 CONSTANT[0]:436
 Function fitting_atoms_backward:
  REG:64 STACK:48 SHARED:25600 LOCAL:0 CONSTANT[0]:428
 Function fitting_atoms_forward:
  REG:72 STACK:0 SHARED:19456 LOCAL:0 CONSTANT[0]:428
"""
    records = checker.parse_resource_usage(text)
    assert set(records) == checker.REQUIRED
    assert records["fitting_atoms_backward"]["stack_bytes"] == 48
    assert all(record["spill_bytes"] == 0 for record in records.values())


def test_benchmark_exposes_jit_prepare_cache_and_steady_memory_fields():
    source = (REPO / "tests/benchmark_nep_fused_fitting.py").read_text()
    for field in (
        "jit_prepare_ms", "jit_cache_hit", "raw_seconds",
        "peak_allocated", "peak_reserved",
    ):
        assert field in source
