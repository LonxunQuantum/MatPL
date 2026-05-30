#!/bin/bash
#SBATCH --job-name=phase2_4_smoke
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/phase2_4_smoke.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "========================================"
echo "Phase 2.4 Smoke Test"
echo "========================================"

# ---- Test 1: Default-off (bit-identical to old behavior) ----
echo ""
echo "=== Test 1: Default-off ==="
rm -rf /tmp/smoke_test1
mkdir -p /tmp/smoke_test1

python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']
del cfg['optimizer']['t_mult']
json.dump(cfg, open('/tmp/smoke_test1/cfg.json', 'w'), indent=2)
"

python main.py train /tmp/smoke_test1/cfg.json 2>&1 | tail -5
RC1=$?
if [ $RC1 -ne 0 ]; then
    echo "FAIL: Test 1 (default-off) failed with exit code $RC1"
    exit 1
fi
echo "PASS: Test 1 (default-off)"

# ---- Test 2: EMA opt-in ----
echo ""
echo "=== Test 2: EMA(0.999) opt-in ==="
rm -rf /tmp/smoke_test2
mkdir -p /tmp/smoke_test2

python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']
del cfg['optimizer']['t_mult']
cfg['optimizer']['ema_decay'] = 0.999
json.dump(cfg, open('/tmp/smoke_test2/cfg.json', 'w'), indent=2)
"

python main.py train /tmp/smoke_test2/cfg.json 2>&1 | tail -5
RC2=$?
if [ $RC2 -ne 0 ]; then
    echo "FAIL: Test 2 (EMA opt-in) failed with exit code $RC2"
    exit 1
fi
echo "PASS: Test 2 (EMA opt-in)"

# ---- Test 3: WSD opt-in ----
echo ""
echo "=== Test 3: WSD opt-in ==="
rm -rf /tmp/smoke_test3
mkdir -p /tmp/smoke_test3

python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']
del cfg['optimizer']['t_mult']
cfg['optimizer']['lr_scheduler'] = 'wsd'
cfg['optimizer']['wsd_stable_frac'] = 0.9
cfg['optimizer']['wsd_decay_kind'] = 'cosine'
json.dump(cfg, open('/tmp/smoke_test3/cfg.json', 'w'), indent=2)
"

python main.py train /tmp/smoke_test3/cfg.json 2>&1 | tail -5
RC3=$?
if [ $RC3 -ne 0 ]; then
    echo "FAIL: Test 3 (WSD opt-in) failed with exit code $RC3"
    exit 1
fi
echo "PASS: Test 3 (WSD opt-in)"

echo ""
echo "========================================"
echo "ALL SMOKE TESTS PASSED"
echo "========================================"
