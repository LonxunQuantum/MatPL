#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_SCRIPT="$PROJECT_ROOT/src/build.sh"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEMP_DIR"' EXIT

FAKE_BIN="$TEMP_DIR/bin"
mkdir -p "$FAKE_BIN"
# Keep detection independent of toolkits installed on the test node.
ln -s "$(command -v bash)" "$FAKE_BIN/bash"
ln -s "$(command -v dirname)" "$FAKE_BIN/dirname"

cat >"$FAKE_BIN/python" <<'PY'
#!/usr/bin/env bash
exit_code=${FAKE_PYTHON_EXIT:-0}
[ "$exit_code" -eq 0 ] || exit "$exit_code"
printf '%s\n' "${FAKE_TORCH_BACKEND:?FAKE_TORCH_BACKEND is required}"
PY

cat >"$FAKE_BIN/nvcc" <<'SH'
#!/usr/bin/env bash
exit "${FAKE_COMPILER_EXIT:-0}"
SH

cp "$FAKE_BIN/nvcc" "$FAKE_BIN/hipcc"
chmod +x "$FAKE_BIN/python" "$FAKE_BIN/nvcc" "$FAKE_BIN/hipcc"

run_dry() {
    local backend=$1
    FAKE_TORCH_BACKEND="$backend" \
    PATH="$FAKE_BIN" \
    CUDACXX="$FAKE_BIN/nvcc" \
    MATPL_DTK_ROOT="$TEMP_DIR/dtk" \
    ROCM_PATH="$TEMP_DIR/rocm" \
    MATPL_DTK_NVCC="$FAKE_BIN/nvcc" \
    MATPL_DTK_CUDA_ROOT="$TEMP_DIR/cuda" \
    MATPL_CUDA_ARCHITECTURES="${TEST_CUDA_ARCHITECTURES:-60;70;75;80;86;89;90}" \
    CUDAToolkit_ROOT="$TEMP_DIR/cuda" \
        "${BUILD_SHELL:-bash}" "$BUILD_SCRIPT" --dry-run -j4
}

assert_contains() {
    local output=$1
    local expected=$2
    if ! grep -Fq -- "$expected" <<<"$output"; then
        echo "Expected output to contain: $expected" >&2
        echo "$output" >&2
        exit 1
    fi
}

assert_not_contains() {
    local output=$1
    local unexpected=$2
    if grep -Fq -- "$unexpected" <<<"$output"; then
        echo "Expected output not to contain: $unexpected" >&2
        echo "$output" >&2
        exit 1
    fi
}

help_output=$(bash "$BUILD_SCRIPT" -h)
assert_contains "$help_output" "Environment variables:"
assert_contains "$help_output" "MATPL_CUDA_ARCHITECTURES"
assert_contains "$help_output" "Default: 60;70;75;80;86;89;90"
assert_contains "$help_output" "V100=70, A100=80, RTX 3090=86"
assert_contains "$help_output" "RTX 4090=89, H20/H100=90"
assert_contains "$help_output" "export MATPL_CUDA_ARCHITECTURES=86"
assert_contains "$help_output" 'export MATPL_CUDA_ARCHITECTURES="70;80;86;90"'

cuda_output=$(run_dry cuda)
assert_contains "$cuda_output" "Resolved accelerator backend: cuda"
assert_contains "$cuda_output" "Operator build backends: cuda cpu"
assert_contains "$cuda_output" "-DCMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89;90"
assert_contains "$cuda_output" "$PROJECT_ROOT/src/op/build/cuda -DMATPL_GPU_BACKEND=CUDA"
assert_contains "$cuda_output" "$PROJECT_ROOT/src/op/build/cpu -DMATPL_GPU_BACKEND=CPU"

# Both CUDA consumers must receive the selected architecture, even on a
# GPU-less host and when unrelated architecture environment variables conflict.
single_arch_output=$(TEST_CUDA_ARCHITECTURES=70 CUDA_VISIBLE_DEVICES= \
    CMAKE_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 run_dry cuda)
assert_contains "$single_arch_output" "$PROJECT_ROOT/src/feature/NEP_GPU/build/cuda"
assert_contains "$single_arch_output" "$PROJECT_ROOT/src/op/build/cuda -DMATPL_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=70"
if [ "$(grep -c -- '-DCMAKE_CUDA_ARCHITECTURES=70' <<<"$single_arch_output")" -ne 2 ]; then
    echo "Expected architecture 70 in both CUDA configure commands" >&2
    exit 1
fi
assert_not_contains "$single_arch_output" "-DCMAKE_CUDA_ARCHITECTURES=86"

multi_arch_output=$(TEST_CUDA_ARCHITECTURES='70;86' run_dry cuda)
assert_contains "$multi_arch_output" "$PROJECT_ROOT/src/op/build/cuda -DMATPL_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=70;86"
assert_not_contains "$(grep -- '-DMATPL_GPU_BACKEND=CPU' <<<"$multi_arch_output")" '-DCMAKE_CUDA_ARCHITECTURES='

hip_output=$(run_dry hip)
assert_contains "$hip_output" "Resolved accelerator backend: hip"
assert_contains "$hip_output" "Operator build backends: hip cpu"
assert_not_contains "$hip_output" "-DCMAKE_CUDA_ARCHITECTURES="
assert_contains "$hip_output" "$PROJECT_ROOT/src/op/build/hip -DMATPL_GPU_BACKEND=HIP"
assert_contains "$hip_output" "$PROJECT_ROOT/src/op/build/cpu -DMATPL_GPU_BACKEND=CPU"

cpu_output=$(run_dry cpu)
assert_contains "$cpu_output" "Resolved accelerator backend: cpu"
assert_contains "$cpu_output" "Operator build backends: cpu"
assert_contains "$cpu_output" "$PROJECT_ROOT/src/op/build/cpu -DMATPL_GPU_BACKEND=CPU"
assert_not_contains "$cpu_output" "$PROJECT_ROOT/src/op/build/cuda"
assert_not_contains "$cpu_output" "$PROJECT_ROOT/src/op/build/hip"

# A CUDA/HIP PyTorch wheel alone is not a compiler environment.
mv "$FAKE_BIN/nvcc" "$TEMP_DIR/nvcc"
cuda_missing_output=$(run_dry cuda)
assert_contains "$cuda_missing_output" "Operator build backends: cpu"
assert_contains "$cuda_missing_output" "nvcc"
assert_not_contains "$cuda_missing_output" "-DMATPL_GPU_BACKEND=CUDA"
assert_not_contains "$cuda_missing_output" "-DMATPL_GPU_BACKEND=HIP"
mv "$TEMP_DIR/nvcc" "$FAKE_BIN/nvcc"

mv "$FAKE_BIN/hipcc" "$TEMP_DIR/hipcc"
hip_missing_output=$(run_dry hip)
assert_contains "$hip_missing_output" "Operator build backends: cpu"
assert_contains "$hip_missing_output" "hipcc"
assert_not_contains "$hip_missing_output" "-DMATPL_GPU_BACKEND=HIP"
assert_not_contains "$hip_missing_output" "-DMATPL_GPU_BACKEND=CUDA"
assert_contains "$hip_missing_output" "Skipping NEP-GPU interface for backend cpu"
mv "$TEMP_DIR/hipcc" "$FAKE_BIN/hipcc"

# A compiler that cannot start (e.g. missing runtime dependencies) is unusable.
for backend in cuda hip; do
    broken_output=$(FAKE_COMPILER_EXIT=1 run_dry "$backend")
    assert_contains "$broken_output" "Operator build backends: cpu"
    assert_not_contains "$broken_output" "-DMATPL_GPU_BACKEND=CUDA"
    assert_not_contains "$broken_output" "-DMATPL_GPU_BACKEND=HIP"
done

# No device visibility check: compilation also works on GPU-less build nodes.
hidden_output=$(CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= run_dry cuda)
assert_contains "$hidden_output" "Operator build backends: cuda cpu"
assert_contains "$hidden_output" "--parallel 4"

# hipcc is sufficient for operators; DTK nvcc is only for the NEP-GPU interface.
mv "$FAKE_BIN/nvcc" "$TEMP_DIR/nvcc"
hip_only_output=$(run_dry hip)
assert_contains "$hip_only_output" "Operator build backends: hip cpu"
assert_contains "$hip_only_output" "Skipping NEP-GPU interface for backend hip"
mv "$TEMP_DIR/nvcc" "$FAKE_BIN/nvcc"

# Preserve the documented `sh build.sh -j4` entry point, including dash systems.
sh_output=$(BUILD_SHELL="$(command -v sh)" run_dry cuda)
assert_contains "$sh_output" "Operator build backends: cuda cpu"
if command -v dash >/dev/null 2>&1; then
    dash_output=$(BUILD_SHELL="$(command -v dash)" run_dry cpu)
    assert_contains "$dash_output" "Operator build backends: cpu"
fi

if FAKE_PYTHON_EXIT=1 run_dry cuda >"$TEMP_DIR/python.out" 2>&1; then
    echo "Expected failed PyTorch import to be rejected" >&2
    exit 1
fi
assert_contains "$(<"$TEMP_DIR/python.out")" "Unable to detect the PyTorch accelerator backend"

if FAKE_TORCH_BACKEND=cuda PATH="$FAKE_BIN:$PATH" \
    bash "$BUILD_SCRIPT" --gpu-backend cuda --dry-run >"$TEMP_DIR/legacy.out" 2>&1; then
    echo "Expected --gpu-backend to be rejected" >&2
    exit 1
fi
assert_contains "$(<"$TEMP_DIR/legacy.out")" "Unknown option --gpu-backend"

# Once a compiler is selected, a real operator build error must stay fatal.
# Use a disposable source tree so the negative test cannot clean real builds.
FIXTURE="$TEMP_DIR/project/src"
mkdir -p "$FIXTURE/op"
cp "$BUILD_SCRIPT" "$FIXTURE/build.sh"
for tool in mkdir rm; do
    ln -s "$(command -v "$tool")" "$FAKE_BIN/$tool"
done
cat >"$FAKE_BIN/cmake" <<'SH'
#!/usr/bin/env bash
exit 42
SH
chmod +x "$FAKE_BIN/cmake"
if FAKE_TORCH_BACKEND=cuda PATH="$FAKE_BIN" \
    bash "$FIXTURE/build.sh" -j4 >"$TEMP_DIR/compile.out" 2>&1; then
    echo "Expected failed operator build to be rejected" >&2
    exit 1
fi
assert_contains "$(<"$TEMP_DIR/compile.out")" "Failed to build operators for backend cuda"
assert_not_contains "$(<"$TEMP_DIR/compile.out")" "MatPL has been successfully installed"

echo "build backend CLI tests passed"
