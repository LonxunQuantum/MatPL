#!/usr/bin/env bash
# Run on a compute node with CUDA PyTorch, nvcc, CMake and pybind11 installed.
# MATPL_TEST_BUILD=1 also compiles both libraries; no GPU is required or used.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf -- "$BUILD_ROOT"' EXIT
export CUDA_VISIBLE_DEVICES=""

check_commands() {
    python - "$1" "$2" <<'PY'
import json
import re
import sys
from pathlib import Path

build_dir = Path(sys.argv[1])
expected = set(sys.argv[2].split(";"))
commands = json.loads((build_dir / "compile_commands.json").read_text())
cuda_commands = [entry for entry in commands if entry["file"].endswith(".cu")]
assert cuda_commands, f"No CUDA commands in {build_dir}"
for entry in cuda_commands:
    actual = set(re.findall(r"(?:compute|sm)_(\d+[a-z]?)", entry["command"]))
    assert actual == expected, (entry["file"], actual, expected, entry["command"])
log = (build_dir / "configure.log").read_text()
assert "Automatic GPU detection failed" not in log, log
print(f"{build_dir.name}: {len(cuda_commands)} CUDA commands target only {sorted(expected)}")
PY
}

check_library() {
    python - "$1" <<'PY'
import re
import subprocess
import sys

listing = subprocess.check_output(["cuobjdump", "--list-elf", sys.argv[1]], text=True)
arches = set(re.findall(r"sm_(\d+)", listing))
assert arches == {"70"}, (sys.argv[1], listing)
print(f"{sys.argv[1]}: compiled device code targets only sm_70")
PY
}

for arch in 70 '70;86'; do
    export MATPL_CUDA_ARCHITECTURES="$arch"
    # MATPL's explicit choice must win over a stale PyTorch architecture list.
    export TORCH_CUDA_ARCH_LIST=8.0
    # Reuse the directory to catch stale cached architecture flags on reconfigure.
    build_dir="$BUILD_ROOT/op"
    mkdir -p "$build_dir"
    cmake -S "$PROJECT_ROOT/src/op" -B "$build_dir" \
        -DMATPL_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=86 \
        2>&1 | tee "$build_dir/configure.log"
    check_commands "$build_dir" "$arch"
    if [ "${MATPL_TEST_BUILD:-0}" = 1 ] && [ "$arch" = 70 ]; then
        cmake --build "$build_dir" --parallel "${MATPL_TEST_JOBS:-4}"
        test -f "$build_dir/lib/libCalcOps_bind.so"
        check_library "$build_dir/lib/libCalcOps_cuda.so"
    fi

    build_dir="$BUILD_ROOT/nep"
    mkdir -p "$build_dir"
    cmake -S "$PROJECT_ROOT/src/feature/NEP_GPU" -B "$build_dir" \
        -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
        2>&1 | tee "$build_dir/configure.log"
    check_commands "$build_dir" "$arch"
    if [ "${MATPL_TEST_BUILD:-0}" = 1 ] && [ "$arch" = 70 ]; then
        cmake --build "$build_dir" --parallel "${MATPL_TEST_JOBS:-4}"
        test -f "$build_dir/nep_gpu.so"
        check_library "$build_dir/nep_gpu.so"
    fi
done

echo "CUDA architecture integration tests passed"
