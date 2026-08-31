#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd "$SRC_DIR/.." && pwd)
BUILD_SCRIPT="$SRC_DIR/build.sh"
SETUP_SCRIPT="$PROJECT_ROOT/deploy/scnet/setup-dcu-env.sh"
INSTALL_SCRIPT="$PROJECT_ROOT/deploy/scnet/install-dcu.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_contains() {
    local haystack=$1
    local needle=$2
    [[ "$haystack" == *"$needle"* ]] || fail "missing output: $needle"
}

test_build_dry_run_routes_hip_nep_gpu_through_dtk_nvcc() {
    local fixture_dir nvcc_path output
    fixture_dir=$(mktemp -d)
    trap 'rm -rf -- "$fixture_dir"' RETURN
    nvcc_path="$fixture_dir/dtk/cuda/cuda-12/bin/nvcc"
    mkdir -p "$(dirname "$nvcc_path")" "$fixture_dir/dtk/cuda/cuda"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$nvcc_path"
    chmod +x "$nvcc_path"

    output=$(MATPL_DTK_ROOT="$fixture_dir/dtk" \
        "$BUILD_SCRIPT" --gpu-backend hip --dry-run -j4)

    assert_contains "$output" "NEP-GPU backend: hip"
    assert_contains "$output" "NEP-GPU CUDA compiler: $nvcc_path"
    assert_contains "$output" "feature/NEP_GPU/build/hip"
    assert_contains "$output" "CUDAToolkit_ROOT=$fixture_dir/dtk/cuda/cuda"
}

test_setup_and_install_scripts_use_parameterized_environment() {
    local fixture_dir module_init conda_init dtk_env nvcc_path output
    fixture_dir=$(mktemp -d)
    trap 'rm -rf -- "$fixture_dir"' RETURN
    module_init="$fixture_dir/module-init.sh"
    conda_init="$fixture_dir/conda.sh"
    dtk_env="$fixture_dir/dtk-env.sh"
    nvcc_path="$fixture_dir/dtk/cuda/cuda-12/bin/nvcc"
    mkdir -p "$(dirname "$nvcc_path")" "$fixture_dir/dtk/cuda/cuda" \
        "$fixture_dir/conda/lib"

    printf '%s\n' \
        'module() {' \
        '  [[ "$1" == "load" && "$2" == "test-gcc-module" ]] || return 1' \
        '  export MATPL_TEST_MODULE_LOADED=1' \
        '}' > "$module_init"
    printf '%s\n' \
        'conda() {' \
        '  [[ "$1" == "activate" && "$2" == "test-conda-env" ]] || return 1' \
        '  export CONDA_PREFIX=$MATPL_TEST_CONDA_PREFIX' \
        '  export MATPL_TEST_CONDA_ACTIVATED=1' \
        '}' > "$conda_init"
    printf '%s\n' 'export MATPL_TEST_DTK_SOURCED=1' > "$dtk_env"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$nvcc_path"
    chmod +x "$nvcc_path"

    MATPL_MODULE_INIT="$module_init" \
        MATPL_GCC_MODULE=test-gcc-module \
        MATPL_CONDA_SH="$conda_init" \
        MATPL_CONDA_ENV=test-conda-env \
        MATPL_TEST_CONDA_PREFIX="$fixture_dir/conda" \
        MATPL_DTK_ROOT="$fixture_dir/dtk" \
        MATPL_DTK_ENV="$dtk_env" \
        MATPL_DTK_NVCC="$nvcc_path" \
        bash -c 'source "$1" >/dev/null; [[ ":${LD_LIBRARY_PATH:-}:" == *":$CONDA_PREFIX/lib:"* ]]' \
        _ "$SETUP_SCRIPT"

    output=$(MATPL_MODULE_INIT="$module_init" \
        MATPL_GCC_MODULE=test-gcc-module \
        MATPL_CONDA_SH="$conda_init" \
        MATPL_CONDA_ENV=test-conda-env \
        MATPL_TEST_CONDA_PREFIX="$fixture_dir/conda" \
        MATPL_DTK_ROOT="$fixture_dir/dtk" \
        MATPL_DTK_ENV="$dtk_env" \
        MATPL_DTK_NVCC="$nvcc_path" \
        MATPL_BUILD_JOBS=3 \
        "$INSTALL_SCRIPT" --dry-run)

    assert_contains "$output" "Resolved operator backend: hip"
    assert_contains "$output" "NEP-GPU CUDA compiler: $nvcc_path"
    assert_contains "$output" "cmake --build"
    assert_contains "$output" "--parallel 3"
}

test_build_dry_run_routes_hip_nep_gpu_through_dtk_nvcc
test_setup_and_install_scripts_use_parameterized_environment
echo "DCU deployment script tests passed"
