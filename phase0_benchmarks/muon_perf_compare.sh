#!/bin/bash
#SBATCH --job-name=muon_perf_compare
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/muon_perf_compare.log

# Performance compare for Fix A (async NaN guard).
#
# Four cells of 1 epoch each on small_Si:
#   1. MUON bare           (no EMA, no WSD, max_norm=2.0)
#   2. MUON + EMA(0.999)   (no WSD)
#   3. MUON + WSD          (no EMA)
#   4. MUON + EMA + WSD    (the slow case the user reported)
#
# Pass condition: cell-4 ``Time avg`` <= 1.10 * cell-1 ``Time avg``.
# Each cell prints "Time avg = X.XXXs/step" parsed from the trainer's
# end-of-epoch summary line.

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

run_cell () {
    local label="$1"
    local cfg_path="$2"
    local logfile="$3"
    echo ""
    echo "==================== $label ===================="
    set -o pipefail
    python main.py train "$cfg_path" 2>&1 | tee "$logfile" | tail -3
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "FAIL: $label exited with $rc"
        return 1
    fi
    # Parse "Training Set: Time X.XXX ..." (printed once at end of epoch)
    local t=$(grep -oE "^Training Set: Time [0-9]+\.[0-9]+" "$logfile" | tail -1 | awk '{print $4}')
    echo "$label time_per_step = $t"
    echo "$label,$t" >> /tmp/muon_perf_compare/summary.csv
}

rm -rf /tmp/muon_perf_compare; mkdir -p /tmp/muon_perf_compare
echo "label,time_per_step_s" > /tmp/muon_perf_compare/summary.csv

# Common base writer
write_cfg () {
    # $1=outpath  $2=add_ema  $3=add_wsd
    python - <<PY
import json, sys
cfg = json.load(open("$BASE_CFG"))
cfg['optimizer']['optimizer'] = 'MUON'
cfg['optimizer']['epochs'] = 1
cfg['optimizer']['print_freq'] = 50
for k in ('t_0', 't_mult'):
    cfg['optimizer'].pop(k, None)
cfg['optimizer'].setdefault('max_norm', 2.0)
cfg['optimizer'].setdefault('norm_type', 2)
cfg['optimizer']['muon_mode'] = 'slice'
cfg['optimizer']['muon_enable_gram'] = True
cfg['optimizer']['muon_flash'] = True
cfg['optimizer']['muon_magma'] = True
if "$2" == "1":
    cfg['optimizer']['ema_decay'] = 0.999
if "$3" == "1":
    cfg['optimizer']['lr_scheduler'] = 'wsd'
    cfg['optimizer']['wsd_stable_frac'] = 0.9
    cfg['optimizer']['wsd_decay_kind'] = 'cosine'
json.dump(cfg, open("$1", 'w'), indent=2)
PY
}

mkdir -p /tmp/muon_perf_compare/c1 /tmp/muon_perf_compare/c2 /tmp/muon_perf_compare/c3 /tmp/muon_perf_compare/c4
write_cfg /tmp/muon_perf_compare/c1/cfg.json 0 0
write_cfg /tmp/muon_perf_compare/c2/cfg.json 1 0
write_cfg /tmp/muon_perf_compare/c3/cfg.json 0 1
write_cfg /tmp/muon_perf_compare/c4/cfg.json 1 1

run_cell "Cell 1: MUON bare"        /tmp/muon_perf_compare/c1/cfg.json /tmp/muon_perf_compare/c1/run.log || exit 1
run_cell "Cell 2: MUON + EMA"       /tmp/muon_perf_compare/c2/cfg.json /tmp/muon_perf_compare/c2/run.log || exit 1
run_cell "Cell 3: MUON + WSD"       /tmp/muon_perf_compare/c3/cfg.json /tmp/muon_perf_compare/c3/run.log || exit 1
run_cell "Cell 4: MUON + EMA + WSD" /tmp/muon_perf_compare/c4/cfg.json /tmp/muon_perf_compare/c4/run.log || exit 1

echo ""
echo "==================== Summary ===================="
cat /tmp/muon_perf_compare/summary.csv

python - <<'PY'
import csv
with open('/tmp/muon_perf_compare/summary.csv') as f:
    rows = list(csv.reader(f))[1:]
times = {label: float(t) for label, t in rows}
bare  = times.get('Cell 1: MUON bare')
combo = times.get('Cell 4: MUON + EMA + WSD')
print(f"\nbare={bare:.4f}s  combo={combo:.4f}s  ratio={combo/bare:.3f}x")
if combo <= 1.10 * bare:
    print("PASS: three-component overhead within 10%.")
else:
    print(f"FAIL: combo is {combo/bare:.2f}x bare (>1.10x). Inspect host syncs / magma path.")
    raise SystemExit(1)
PY
RC=$?
exit $RC
