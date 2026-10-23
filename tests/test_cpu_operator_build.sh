#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BUILD_DIR=$(mktemp -d)
trap 'rm -rf -- "$BUILD_DIR"' EXIT

cmake -S "$PROJECT_ROOT/src/op" -B "$BUILD_DIR" \
    -DMATPL_GPU_BACKEND=CPU
cmake --build "$BUILD_DIR" --target CalcOps_bind_cpu --parallel "${MATPL_TEST_JOBS:-1}"

test -f "$BUILD_DIR/lib/libCalcOps_bind_cpu.so"
test -f "$BUILD_DIR/lib/libCalcOps_bind.so"
test -f "$BUILD_DIR/lib/libCalcOps_cuda.so"

echo "CPU operator build test passed"
