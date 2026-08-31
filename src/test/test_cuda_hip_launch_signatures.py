from pathlib import Path
import re
import unittest


SRC_ROOT = Path(__file__).resolve().parents[1]
OP_ROOT = SRC_ROOT / "op"


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def normalize_parameters(parameters: str):
    normalized = []
    for parameter in parameters.split(","):
        parameter = parameter.split("=", 1)[0].strip()
        if not parameter or parameter == "void":
            continue
        parameter = re.sub(r"\b[A-Za-z_]\w*\s*$", "", parameter).strip()
        normalized.append(re.sub(r"\s+", "", parameter))
    return tuple(normalized)


def extract_launch_definitions(root: Path, suffix: str):
    definitions = {}
    pattern = re.compile(
        r"\bvoid\s+(launch_[A-Za-z0-9_]+)\s*\((.*?)\)\s*\{",
        flags=re.DOTALL,
    )
    for path in sorted(root.rglob(f"*{suffix}")):
        source = strip_comments(path.read_text())
        for match in pattern.finditer(source):
            definitions.setdefault(
                match.group(1), normalize_parameters(match.group(2))
            )
    return definitions


def declared_launch_names():
    names = set()
    pattern = re.compile(r"\bvoid\s+(launch_[A-Za-z0-9_]+)\s*\(")
    for path in sorted((OP_ROOT / "include").glob("*.h")):
        names.update(pattern.findall(strip_comments(path.read_text())))
    return {name for name in names if not name.endswith("_cpu")}


class CudaHipLaunchSignatureTests(unittest.TestCase):
    def test_public_launch_boundaries_match_between_cuda_and_hip(self):
        public_names = declared_launch_names()
        cuda_definitions = extract_launch_definitions(OP_ROOT / "kernel", ".cu")
        hip_definitions = extract_launch_definitions(OP_ROOT / "kernel_hip", ".hip")

        missing_cuda = sorted(public_names - cuda_definitions.keys())
        missing_hip = sorted(public_names - hip_definitions.keys())
        self.assertEqual(missing_cuda, [], "Public launch functions missing from CUDA")
        self.assertEqual(missing_hip, [], "Public launch functions missing from HIP")

        mismatches = {
            name: (cuda_definitions[name], hip_definitions[name])
            for name in sorted(public_names)
            if cuda_definitions[name] != hip_definitions[name]
        }
        self.assertEqual(mismatches, {})

    def test_nep_virial_gradient_launch_is_batch_aware(self):
        hip_source = strip_comments(
            (OP_ROOT / "kernel_hip/calculateNepVirial.hip").read_text()
        )
        match = re.search(
            r"\bvoid\s+launch_calculate_nepvirial_grad\s*\((.*?)\)\s*\{",
            hip_source,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(match)
        self.assertRegex(match.group(1), r"\bnum_atom\b")
        self.assertRegex(match.group(1), r"\bbatch_num\b")


if __name__ == "__main__":
    unittest.main()
