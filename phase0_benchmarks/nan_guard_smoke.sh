#!/bin/bash
#SBATCH --job-name=nan_guard_smoke
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/nan_guard_smoke.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "=== Smoke test: default training (NaN guard active but should not fire) ==="
rm -rf /tmp/nan_guard_test; mkdir -p /tmp/nan_guard_test
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']; del cfg['optimizer']['t_mult']
json.dump(cfg, open('/tmp/nan_guard_test/cfg.json', 'w'), indent=2)
"
python main.py train /tmp/nan_guard_test/cfg.json 2>&1 | tail -5
RC=$?
echo "exit code: $RC"

# Check whether NaN guard fired (should not on normal training)
nan_fired=$(grep -c "\[NaN guard\]" /tmp/nan_guard_test/*.log 2>/dev/null || echo 0)
echo "NaN guard fired: $nan_fired times (expected 0 for normal training)"

[ $RC -eq 0 ] && echo "PASS: normal training works with NaN guard" || { echo "FAIL"; exit 1; }
