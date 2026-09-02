#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_SCRIPT="$PROJECT_ROOT/src/build.sh"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEMP_DIR"' EXIT

FAKE_BIN="$TEMP_DIR/bin"
mkdir -p "$FAKE_BIN"

cat >"$FAKE_BIN/python" <<'PY'
#!/usr/bin/env bash
printf '%s\n' "${FAKE_TORCH_BACKEND:?FAKE_TORCH_BACKEND is required}"
PY

cat >"$FAKE_BIN/nvcc" <<'SH'
#!/usr/bin/env bash
exit 0
SH

chmod +x "$FAKE_BIN/python" "$FAKE_BIN/nvcc"

run_dry() {
    local backend=$1
    FAKE_TORCH_BACKEND="$backend" \
    PATH="$FAKE_BIN:$PATH" \
    MATPL_DTK_NVCC="$FAKE_BIN/nvcc" \
    MATPL_DTK_CUDA_ROOT="$TEMP_DIR/cuda" \
    CUDAToolkit_ROOT="$TEMP_DIR/cuda" \
        bash "$BUILD_SCRIPT" --dry-run -j2
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

if FAKE_TORCH_BACKEND=cuda PATH="$FAKE_BIN:$PATH" \
    bash "$BUILD_SCRIPT" --gpu-backend cuda --dry-run >"$TEMP_DIR/legacy.out" 2>&1; then
    echo "Expected --gpu-backend to be rejected" >&2
    exit 1
fi
assert_contains "$(<"$TEMP_DIR/legacy.out")" "Unknown option --gpu-backend"

echo "build backend CLI tests passed"
