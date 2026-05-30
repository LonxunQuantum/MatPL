#!/bin/bash
#SBATCH -p 4090
#SBATCH -J MatPL_bench
#SBATCH -N 1
#SBATCH -o bench_out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

# Phase 1 benchmark: run actual training (no profiler overhead) for 1 epoch
# and extract per-step time from the training log.
#
# Usage: sbatch run_bench_noprof.sh   OR   bash run_bench_noprof.sh

set -e
export PYTHONPATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL:$PYTHONPATH
export PATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/src/bin:$PATH

MATPL_ROOT=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL
BENCH_DIR=$MATPL_ROOT/phase0_benchmarks

run_train() {
    local tag=$1
    local json="$BENCH_DIR/${tag}.json"
    local logfile="$BENCH_DIR/train_${tag}.log"

    echo "==========================================="
    echo "[$(date '+%H:%M:%S')] Training benchmark: $tag (no profiler)"
    echo "==========================================="

    cd "$(dirname "$json")"
    python "$MATPL_ROOT/main.py" train "$json" 2>&1 | tee "$logfile"

    # Extract average step time from last 50 lines
    echo
    echo "--- $tag: last 10 reported Time values ---"
    grep "^Epoch:" "$logfile" | tail -10 | awk '{for(i=1;i<=NF;i++) if($i=="Time") print $(i+1)}'
    echo
}

run_train small_Si
rm -rf model_record
run_train medium_C
rm -rf model_record
# large_water runs 48419 steps per epoch — too long for quick bench.
# Run only 500 steps by using nep_profile without the profiler context.
# Instead just time the first 500 lines of output.
echo "==========================================="
echo "[$(date '+%H:%M:%S')] large_water: timing first 500 steps from train log"
echo "==========================================="
cd "$BENCH_DIR"
timeout 120 python "$MATPL_ROOT/main.py" train "$BENCH_DIR/large_water.json" 2>&1 | head -600 > "$BENCH_DIR/train_large_water.log" || true
echo "--- large_water: last 10 reported Time values ---"
grep "^Epoch:" "$BENCH_DIR/train_large_water.log" | tail -10 | awk '{for(i=1;i<=NF;i++) if($i=="Time") print $(i+1)}'

echo
echo "Done. Compare Time column with baseline (Si ~0.019s, C ~0.036s, water ~0.032s per step)."
