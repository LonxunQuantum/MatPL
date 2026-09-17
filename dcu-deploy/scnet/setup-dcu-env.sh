#!/usr/bin/env bash

# This file must be sourced so that the compiler and runtime environment remain
# active in the caller's shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Error: source this script instead of executing it:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

MATPL_MODULE_INIT=${MATPL_MODULE_INIT:-/etc/profile}
MATPL_GCC_MODULE=${MATPL_GCC_MODULE:-compiler/gcc/9.3.0}
MATPL_PYTHON_ACTIVATE=${MATPL_PYTHON_ACTIVATE:-/public/home/suolaoliu/tools/app/conda/matpl-2026.3/bin/activate}
MATPL_DTK_ROOT=${MATPL_DTK_ROOT:-/public/software/compiler/dtk-26.04}
MATPL_DTK_ENV=${MATPL_DTK_ENV:-$MATPL_DTK_ROOT/env.sh}

if [[ ! -r "$MATPL_MODULE_INIT" ]]; then
    echo "Error: module initialization script not found: $MATPL_MODULE_INIT" >&2
    return 1
fi
source "$MATPL_MODULE_INIT"

if ! command -v module >/dev/null 2>&1; then
    echo "Error: environment modules are unavailable after sourcing $MATPL_MODULE_INIT" >&2
    return 1
fi
module load "$MATPL_GCC_MODULE" || return 1

if [[ ! -r "$MATPL_PYTHON_ACTIVATE" ]]; then
    echo "Error: Python environment activation script not found: $MATPL_PYTHON_ACTIVATE" >&2
    return 1
fi
source "$MATPL_PYTHON_ACTIVATE" || return 1
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Error: Conda activation did not set CONDA_PREFIX" >&2
    return 1
fi
case ":${LD_LIBRARY_PATH:-}:" in
    *":$CONDA_PREFIX/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

if [[ ! -r "$MATPL_DTK_ENV" ]]; then
    echo "Error: DTK environment script not found: $MATPL_DTK_ENV" >&2
    return 1
fi
# Do not enable nounset before this point: DTK's environment script reads
# optional variables such as CMAKE_PREFIX_PATH.
source "$MATPL_DTK_ENV"

if [[ -z "${MATPL_DTK_NVCC:-}" ]]; then
    for candidate in \
        "$MATPL_DTK_ROOT"/cuda/cuda-*/bin/nvcc \
        "$MATPL_DTK_ROOT/cuda/cuda/bin/nvcc"; do
        if [[ -x "$candidate" ]]; then
            MATPL_DTK_NVCC=$candidate
            break
        fi
    done
fi

if [[ -z "${MATPL_DTK_NVCC:-}" || ! -x "$MATPL_DTK_NVCC" ]]; then
    echo "Error: DTK nvcc-compatible compiler was not found under $MATPL_DTK_ROOT" >&2
    echo "Set MATPL_DTK_NVCC to its absolute path and source this script again." >&2
    return 1
fi

export MATPL_DTK_ROOT MATPL_DTK_NVCC
export MATPL_DTK_CUDA_ROOT=${MATPL_DTK_CUDA_ROOT:-$MATPL_DTK_ROOT/cuda/cuda}
export CUDACXX=$MATPL_DTK_NVCC
export CUDAToolkit_ROOT=$MATPL_DTK_CUDA_ROOT
case ":$PATH:" in
    *":$(dirname "$MATPL_DTK_NVCC"):"*) ;;
    *) export PATH="$(dirname "$MATPL_DTK_NVCC"):$PATH" ;;
esac

echo "MatPL DCU environment ready:"
echo "  Python environment: $CONDA_PREFIX"
echo "  DTK root: $MATPL_DTK_ROOT"
echo "  CUDA-compatible compiler: $MATPL_DTK_NVCC"
