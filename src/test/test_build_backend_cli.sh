#!/bin/bash
set -euo pipefail

SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FAKE_BIN=$(mktemp -d)
trap 'rm -rf "$FAKE_BIN"' EXIT

printf '%s\n' \
    '#!/bin/sh' \
    'printf "%s\\n" "${MATPL_TEST_TORCH_BACKEND:-hip}"' \
    > "$FAKE_BIN/python"
chmod +x "$FAKE_BIN/python"

assert_contains() {
    haystack=$1
    needle=$2
    if [[ "$haystack" != *"$needle"* ]]; then
        printf 'Expected output to contain: %s\nActual output:\n%s\n' \
            "$needle" "$haystack" >&2
        exit 1
    fi
}

run_build() {
    (
        cd "$SRC_DIR"
        PATH="$FAKE_BIN:$PATH" bash ./build.sh "$@"
    )
}

output=$(MATPL_TEST_TORCH_BACKEND=hip run_build \
    --dry-run --gpu-backend auto -j4 -m nn)
assert_contains "$output" "Requested operator backend: auto"
assert_contains "$output" "Resolved operator backend: hip"
assert_contains "$output" "op/build/hip"
assert_contains "$output" "-DMATPL_GPU_BACKEND=HIP"
assert_contains "$output" "--parallel 4"
assert_contains "$output" "Compile Fortran codes: Yes"

output=$(run_build --dry-run --gpu-backend CUDA -j2)
assert_contains "$output" "Resolved operator backend: cuda"
assert_contains "$output" "op/build/cuda"
assert_contains "$output" "--parallel 2"

output=$(run_build --dry-run --gpu-backend=cpu)
assert_contains "$output" "Resolved operator backend: cpu"
assert_contains "$output" "op/build/cpu"

if run_build --dry-run --gpu-backend metal > /dev/null 2>&1; then
    echo "Unsupported backend was accepted" >&2
    exit 1
fi

echo "build.sh backend CLI tests passed"
