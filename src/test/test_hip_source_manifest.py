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

    def test_hip_force_launches_fit_dtk_kernel_bounds(self):
        oversized = []
        launch_pattern = re.compile(
            r"(?:(?:const\s+)?int\s+)?LEN\s*=\s*(\d+)\s*;"
            r"(?:(?!LEN\s*=).){0,240}?dim3\s+thread_grid_?\s*\(\s*LEN\s*,\s*([34])\s*\)",
            flags=re.DOTALL,
        )
        for filename in ("calculateForce.hip", "calculateNepForce.hip"):
            source = (HIP_ROOT / filename).read_text()
            for match in launch_pattern.finditer(source):
                threads = int(match.group(1)) * int(match.group(2))
                if threads > 256:
                    oversized.append(f"{filename}:{threads}")

        self.assertEqual(oversized, [])

    def test_hip_virial_backward_launch_fits_dtk_kernel_bounds(self):
        source = (HIP_ROOT / "calculateNepVirial.hip").read_text()
        launch = re.search(
            r"launch_calculate_nepvirial_grad\s*\([^)]*\)\s*\{"
            r"(?:(?!^}).)*?LEN\s*=\s*(\d+)\s*;"
            r"(?:(?!^}).)*?dim3\s+thread_grid\s*\(\s*LEN\s*,\s*4\s*\)",
            source,
            flags=re.DOTALL | re.MULTILINE,
        )

        self.assertIsNotNone(launch, "HIP virial backward launch was not found")
        self.assertLessEqual(int(launch.group(1)) * 4, 256)


if __name__ == "__main__":
    unittest.main()
