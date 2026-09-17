#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SETUP_SCRIPT="$PROJECT_ROOT/dcu-deploy/scnet/setup-dcu-env.sh"

fixture_dir=$(mktemp -d)
trap 'rm -rf -- "$fixture_dir"' EXIT

module_init="$fixture_dir/module-init.sh"
python_activate="$fixture_dir/python-env/bin/activate"
dtk_env="$fixture_dir/dtk/env.sh"
nvcc_path="$fixture_dir/dtk/cuda/cuda-12/bin/nvcc"

mkdir -p "$(dirname "$python_activate")" "$(dirname "$dtk_env")" \
    "$(dirname "$nvcc_path")" "$fixture_dir/python-env/lib" \
    "$fixture_dir/dtk/cuda/cuda"

printf '%s\n' \
    'module() {' \
    '  [[ "$1" == "load" && "$2" == "test-gcc-module" ]] || return 1' \
    '  export MATPL_TEST_MODULE_LOADED=1' \
    '}' > "$module_init"
printf '%s\n' \
    'export CONDA_PREFIX=$MATPL_TEST_CONDA_PREFIX' \
    'export MATPL_TEST_PYTHON_ACTIVATED=1' > "$python_activate"
printf '%s\n' 'export MATPL_TEST_DTK_SOURCED=1' > "$dtk_env"
printf '#!/usr/bin/env bash\nexit 0\n' > "$nvcc_path"
chmod +x "$nvcc_path"

MATPL_MODULE_INIT="$module_init" \
    MATPL_GCC_MODULE=test-gcc-module \
    MATPL_PYTHON_ACTIVATE="$python_activate" \
    MATPL_TEST_CONDA_PREFIX="$fixture_dir/python-env" \
    MATPL_DTK_ROOT="$fixture_dir/dtk" \
    MATPL_DTK_ENV="$dtk_env" \
    MATPL_DTK_NVCC="$nvcc_path" \
    bash -c '
        source "$1" >/dev/null
        [[ "${MATPL_TEST_MODULE_LOADED:-}" == 1 ]]
        [[ "${MATPL_TEST_PYTHON_ACTIVATED:-}" == 1 ]]
        [[ "${MATPL_TEST_DTK_SOURCED:-}" == 1 ]]
        [[ ":${LD_LIBRARY_PATH:-}:" == *":$CONDA_PREFIX/lib:"* ]]
        [[ "$CUDACXX" == "$MATPL_DTK_NVCC" ]]
    ' _ "$SETUP_SCRIPT"

echo "DCU deployment script tests passed"
