#!/bin/bash
#SBATCH --job-name=muon_smoke
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/muon_smoke.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "=== Phase 3.3 smoke test: HybridMuon on small_Si ==="
rm -rf /tmp/muon_test; mkdir -p /tmp/muon_test
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['optimizer'] = 'MUON'
cfg['optimizer']['epochs'] = 2
cfg['optimizer']['print_freq'] = 10
# Muon doesn't use CosineAnnealingWarmRestarts; remove t_0/t_mult.
# KEEP max_norm/norm_type -- Phase 3.3 gate now requires gradient clipping for MUON.
for k in ('t_0', 't_mult'):
    cfg['optimizer'].pop(k, None)
cfg['optimizer'].setdefault('max_norm', 2.0)
cfg['optimizer'].setdefault('norm_type', 2)
# Set muon-specific knobs
cfg['optimizer']['muon_mode'] = 'slice'
cfg['optimizer']['muon_enable_gram'] = True
cfg['optimizer']['muon_flash'] = True
cfg['optimizer']['muon_magma'] = True
cfg['optimizer']['muon_lr_adjust'] = 0.0
cfg['optimizer']['muon_lr_adjust_coeff'] = 0.18
json.dump(cfg, open('/tmp/muon_test/cfg.json', 'w'), indent=2)
print('Config written with optimizer=MUON')
"

set -o pipefail
python main.py train /tmp/muon_test/cfg.json 2>&1 | tee /tmp/muon_test/run.log | tail -30
RC=${PIPESTATUS[0]}
echo "exit code: $RC"

# Check for NaN guard
nan_fired=$(grep -c "\[NaN guard\]" /tmp/muon_test/run.log 2>/dev/null || echo 0)
echo "NaN guard fired: $nan_fired times"

[ $RC -eq 0 ] && echo "PASS: HybridMuon smoke test passed" || { echo "FAIL"; exit 1; }
