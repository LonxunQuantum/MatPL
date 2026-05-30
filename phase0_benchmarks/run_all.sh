#!/bin/bash
#SBATCH -p 4090
#SBATCH -J MatPL
#SBATCH -N 1
#SBATCH -o out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

# Phase 0 baseline runner for MatPL NEP training.
#
# Usage:
#   bash run_all.sh             # run all three benchmarks
#   bash run_all.sh small       # run only small_Si
#   bash run_all.sh medium      # run only medium_C
#   bash run_all.sh large       # run only large_water
#
# Assumes:
#   - Run on a node with CUDA available (NOT the login node).
#   - The MatPL env script can be sourced: matpl-env.sh
#   - CalcOps libCalcOps_bind.so has been built under MatPL/src/op/build/

set -e
export PYTHONPATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL:$PYTHONPATH
export PATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/src/bin:$PATH
#ENV_SCRIPT=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-env.sh
MATPL_ROOT=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL
BENCH_DIR=$MATPL_ROOT/phase0_benchmarks

# shellcheck disable=SC1090
#source "$ENV_SCRIPT"

# Default: 200 steps with 5 warmup + 10 active (covers >1 profile window).
STEPS=${MATPL_PROFILE_STEPS:-200}
WARMUP=${MATPL_PROFILE_WARMUP:-5}
ACTIVE=${MATPL_PROFILE_ACTIVE:-10}

run_one() {
    local tag=$1
    local json="$BENCH_DIR/${tag}.json"
    local out_dir="$BENCH_DIR/out_${tag}"

    if [ ! -f "$json" ]; then
        echo "[SKIP] $json not found"
        return
    fi

    echo "==========================================="
    echo "[$(date '+%H:%M:%S')] running benchmark: $tag"
    echo "  json    : $json"
    echo "  out_dir : $out_dir"
    echo "  steps   : $STEPS (warmup=$WARMUP active=$ACTIVE)"
    echo "==========================================="

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    cd "$(dirname "$json")"

    python "$MATPL_ROOT/main.py" nep_profile "$json" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --active "$ACTIVE" \
        --memsnap \
        --out "$out_dir" 2>&1 | tee "$out_dir/run.log"

    python "$MATPL_ROOT/src/utils/profile_report.py" "$out_dir" || true

    echo "[$(date '+%H:%M:%S')] finished: $tag"
    echo
}

case "${1:-all}" in
    small)  run_one small_Si ;;
    medium) run_one medium_C ;;
    large)  run_one large_water ;;
    all)
        run_one small_Si
        run_one medium_C
        run_one large_water
        ;;
    *)
        echo "Unknown target: $1 (valid: small | medium | large | all)"
        exit 1
        ;;
esac

echo "All requested benchmarks done. Reports under $BENCH_DIR/out_*/phase0_report.md"
