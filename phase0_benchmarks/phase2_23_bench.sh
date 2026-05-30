#!/bin/bash
#SBATCH --job-name=phase2_23_bench
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/phase2_23_bench.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BENCH_DIR="$CODE_DIR/phase0_benchmarks"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

run_one() {
    local label="$1"
    local base_cfg="$2"
    local same_nloc="$3"   # True / False (Python bool)
    local compile_f="$4"   # True / False (Python bool)
    local outdir="/tmp/p23_bench_${label}"
    rm -rf "$outdir"; mkdir -p "$outdir"

    python -c "
import json
cfg = json.load(open('$base_cfg'))
cfg['optimizer']['epochs'] = 1
if 't_0' in cfg['optimizer']: del cfg['optimizer']['t_0']
if 't_mult' in cfg['optimizer']: del cfg['optimizer']['t_mult']
cfg['same_nloc_sampler'] = $same_nloc
cfg['compile_fitting'] = $compile_f
cfg['model_store_dir'] = '$outdir'
json.dump(cfg, open('$outdir/cfg.json', 'w'), indent=2)
"
    echo "================ $label (sampler=$same_nloc, compile=$compile_f) ================"
    timeout 600 python main.py train "$outdir/cfg.json" 2>&1 | tee "$outdir/train.log" | tail -3
    avg=$(grep "Epoch:" "$outdir/train.log" | tail -50 | sed -nE 's/.*Time +[0-9.]+ +\( *([0-9.]+).*/\1/p' | awk '{s+=$1; n++} END{if(n)print s/n; else print "n/a"}')
    echo ">>> $label avg s/step = $avg"
    echo ""
}

# medium_C: ntypes=1, network=[100,1], reasonable dataset
echo "######## medium_C (ntypes=1, net=[100,1]) ########"
run_one medC_baseline       "$BENCH_DIR/medium_C.json"     False False
run_one medC_bucket         "$BENCH_DIR/medium_C.json"     True  False
run_one medC_compile        "$BENCH_DIR/medium_C.json"     False True
run_one medC_bucket_compile "$BENCH_DIR/medium_C.json"     True  True

# large_water: ntypes=2, network=[100,1] — best fit for compile
echo "######## large_water (ntypes=2, net=[100,1]) ########"
run_one wat_baseline       "$BENCH_DIR/large_water.json"  False False
run_one wat_bucket         "$BENCH_DIR/large_water.json"  True  False
run_one wat_compile        "$BENCH_DIR/large_water.json"  False True
run_one wat_bucket_compile "$BENCH_DIR/large_water.json"  True  True

echo ""
echo "##########################################################"
echo "RESULTS SUMMARY"
echo "##########################################################"
for label in medC_baseline medC_bucket medC_compile medC_bucket_compile \
             wat_baseline wat_bucket wat_compile wat_bucket_compile; do
    log="/tmp/p23_bench_${label}/train.log"
    if [ -f "$log" ]; then
        avg=$(grep "Epoch:" "$log" | tail -50 | sed -nE 's/.*Time +[0-9.]+ +\( *([0-9.]+).*/\1/p' | awk '{s+=$1; n++} END{if(n)print s/n; else print "n/a"}')
        echo "$label: avg=$avg s/step"
    fi
done
