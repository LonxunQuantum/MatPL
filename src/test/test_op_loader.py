from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from src.utils.op_loader import (
    detect_compiled_backend,
    get_library_path,
    load_calc_ops,
    select_runtime_backend,
)


class FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


def make_torch(*, hip=None, cuda_version=None, gpu_available=False):
    loaded = []
    cuda_namespace = object()
    cpu_namespace = object()
    ops = SimpleNamespace(
        CalcOps_cuda=cuda_namespace,
        CalcOps_cpu=cpu_namespace,
        load_library=loaded.append,
    )
    module = SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda_version),
        cuda=FakeCuda(gpu_available),
        ops=ops,
    )
    return module, loaded, cuda_namespace, cpu_namespace


class BackendDetectionTests(unittest.TestCase):
    def test_hip_build_wins_when_torch_exposes_both_compatibility_versions(self):
        torch_module, _, _, _ = make_torch(
            hip="6.3", cuda_version="12.4", gpu_available=True
        )

        self.assertEqual(detect_compiled_backend(torch_module), "hip")

    def test_cuda_build_is_detected_from_torch_cuda_version(self):
        torch_module, _, _, _ = make_torch(
            cuda_version="12.4", gpu_available=True
        )

        self.assertEqual(detect_compiled_backend(torch_module), "cuda")

    def test_cpu_build_is_detected_when_torch_has_no_gpu_version(self):
        torch_module, _, _, _ = make_torch()

        self.assertEqual(detect_compiled_backend(torch_module), "cpu")

    def test_runtime_falls_back_to_cpu_when_no_gpu_is_available(self):
        torch_module, _, _, _ = make_torch(
            hip="6.3", cuda_version="12.4", gpu_available=False
        )

        self.assertEqual(select_runtime_backend(torch_module), "cpu")

    def test_runtime_uses_compiled_gpu_backend_when_gpu_is_available(self):
        hip_torch, _, _, _ = make_torch(hip="6.3", gpu_available=True)
        cuda_torch, _, _, _ = make_torch(cuda_version="12.4", gpu_available=True)

        self.assertEqual(select_runtime_backend(hip_torch), "hip")
        self.assertEqual(select_runtime_backend(cuda_torch), "cuda")


class LibraryPathTests(unittest.TestCase):
    def test_each_backend_uses_its_isolated_build_tree(self):
        src_root = Path("/workspace/src")

        self.assertEqual(
            get_library_path("cuda", src_root),
            src_root / "op/build/cuda/lib/libCalcOps_bind.so",
        )
        self.assertEqual(
            get_library_path("hip", src_root),
            src_root / "op/build/hip/lib/libCalcOps_bind.so",
        )
        self.assertEqual(
            get_library_path("cpu", src_root),
            src_root / "op/build/cpu/lib/libCalcOps_bind_cpu.so",
        )

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CalcOps backend"):
            get_library_path("metal", Path("/workspace/src"))


class LibraryLoadingTests(unittest.TestCase):
    def _create_library(self, src_root: Path, backend: str) -> Path:
        library_path = get_library_path(backend, src_root)
        library_path.parent.mkdir(parents=True)
        library_path.touch()
        return library_path

    def test_gpu_library_is_loaded_once_and_returns_gpu_namespace(self):
        torch_module, loaded, cuda_namespace, _ = make_torch(
            hip="6.3", gpu_available=True
        )
        with TemporaryDirectory() as temporary_directory:
            src_root = Path(temporary_directory) / "src"
            library_path = self._create_library(src_root, "hip")

            first = load_calc_ops(torch_module, src_root)
            second = load_calc_ops(torch_module, src_root)

        self.assertIs(first, cuda_namespace)
        self.assertIs(second, cuda_namespace)
        self.assertEqual(loaded, [str(library_path)])

    def test_cpu_fallback_loads_cpu_library_and_namespace(self):
        torch_module, loaded, _, cpu_namespace = make_torch(
            cuda_version="12.4", gpu_available=False
        )
        with TemporaryDirectory() as temporary_directory:
            src_root = Path(temporary_directory) / "src"
            library_path = self._create_library(src_root, "cpu")

            namespace = load_calc_ops(torch_module, src_root)

        self.assertIs(namespace, cpu_namespace)
        self.assertEqual(loaded, [str(library_path)])

    def test_missing_library_error_names_backend_path_and_build_command(self):
        torch_module, _, _, _ = make_torch(hip="6.3", gpu_available=True)
        with TemporaryDirectory() as temporary_directory:
            src_root = Path(temporary_directory) / "src"

            with self.assertRaises(FileNotFoundError) as context:
                load_calc_ops(torch_module, src_root)

        message = str(context.exception)
        self.assertIn("hip", message)
        self.assertIn("libCalcOps_bind.so", message)
        self.assertIn("--gpu-backend hip", message)


if __name__ == "__main__":
    unittest.main()
