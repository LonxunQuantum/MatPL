from pathlib import Path
import re
import unittest


SRC_ROOT = Path(__file__).resolve().parents[1]
OP_ROOT = SRC_ROOT / "op"
CUDA_ROOT = OP_ROOT / "kernel"
HIP_ROOT = OP_ROOT / "kernel_hip"


class HipSourceManifestTests(unittest.TestCase):
    def test_each_cuda_operator_has_an_isolated_hip_translation(self):
        cuda_operators = {path.stem for path in CUDA_ROOT.glob("*.cu")}
        hip_operators = {path.stem for path in HIP_ROOT.glob("*.hip")}

        self.assertTrue(cuda_operators, "CUDA operator manifest is empty")
        self.assertEqual(hip_operators, cuda_operators)

    def test_required_hip_utility_sources_are_present(self):
        required = {
            "common.cuh",
            "error.cuh",
            "error.hip",
            "gpu_vector.cuh",
            "gpu_vector.hip",
            "main_common.cuh",
            "main_common.hip",
            "nep3_small_box.cuh",
            "nep3_small_box_mbgrad.cuh",
            "nep_utilities.cuh",
            "nep_utilities_mb_secondc.cuh",
        }
        actual = {
            path.name
            for path in (HIP_ROOT / "utilities").glob("*")
            if path.is_file()
        }

        self.assertEqual(actual, required)

    def test_cuda_and_hip_cmake_inputs_do_not_cross_backend_trees(self):
        cuda_cmake = (OP_ROOT / "cmake/cuda/CMakeLists.txt").read_text()
        hip_cmake = (OP_ROOT / "cmake/hip/CMakeLists.txt").read_text()

        self.assertIn("kernel/*.cu", cuda_cmake)
        self.assertNotIn("kernel_hip", cuda_cmake)
        self.assertIn("kernel_hip/*.hip", hip_cmake)
        self.assertNotRegex(hip_cmake, r"(?<!_)kernel/\*\.cu")

    def test_hip_sources_do_not_call_cuda_runtime_functions(self):
        direct_cuda_call = re.compile(
            r"\bcuda(?:SetDevice|GetLastError|DeviceSynchronize|Malloc|Memcpy)\s*\("
        )
        violations = []
        for path in sorted(HIP_ROOT.rglob("*")):
            if path.suffix not in {".hip", ".cuh"}:
                continue
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if direct_cuda_call.search(line):
                    violations.append(f"{path.relative_to(OP_ROOT)}:{line_number}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
