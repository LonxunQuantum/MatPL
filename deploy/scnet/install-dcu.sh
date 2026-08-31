#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MATPL_BUILD_JOBS=${MATPL_BUILD_JOBS:-4}

if ! [[ "$MATPL_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MATPL_BUILD_JOBS must be a positive integer" >&2
    exit 1
fi

source "$SCRIPT_DIR/setup-dcu-env.sh"

"$PROJECT_ROOT/src/build.sh" \
    "$@" \
    --gpu-backend hip \
    "-j$MATPL_BUILD_JOBS"
