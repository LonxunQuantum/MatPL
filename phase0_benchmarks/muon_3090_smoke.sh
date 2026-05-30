#!/bin/bash
#SBATCH --job-name=muon_3090_smoke
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/muon_3090_smoke.log

# Compatibility smoke for HybridMuon on Ampere sm_86 (RTX 3090).
#
# Tests two scenarios:
#   1. muon_compile_gram=false (default after Fix B) -> must PASS.
#      Confirms the eager Gram orthogonalizer runs cleanly on sm_86.
#   2. muon_compile_gram=true -> ALLOWED to fail with
#      "device kernel image is invalid", logged for the record so future
#      regressions on this stack are easy to triage.

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "============================================================"
echo " 3090 (sm_86) compatibility smoke test for HybridMuon"
echo " GPU info:"
python -c "import torch; print('   torch=', torch.__version__); print('   capability=', torch.cuda.get_device_capability(0)); print('   name=', torch.cuda.get_device_name(0))"
echo "============================================================"

# ---- Scenario 1: muon_compile_gram=false (default) ----
echo ""
echo "=== Scenario 1: default (muon_compile_gram=false) — MUST PASS ==="
rm -rf /tmp/muon_3090_default; mkdir -p /tmp/muon_3090_default
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['optimizer'] = 'MUON'
cfg['optimizer']['epochs'] = 1
cfg['optimizer']['print_freq'] = 50
for k in ('t_0', 't_mult'):
    cfg['optimizer'].pop(k, None)
# MUON gate: explicit clipping required
cfg['optimizer'].setdefault('max_norm', 2.0)
cfg['optimizer'].setdefault('norm_type', 2)
# Fix B default: compile_gram defaults to false (do not set explicitly).
cfg['optimizer']['muon_mode'] = 'slice'
cfg['optimizer']['muon_enable_gram'] = True
cfg['optimizer']['muon_flash'] = True
cfg['optimizer']['muon_magma'] = True
json.dump(cfg, open('/tmp/muon_3090_default/cfg.json', 'w'), indent=2)
print('Config written for Scenario 1.')
"

set -o pipefail
python main.py train /tmp/muon_3090_default/cfg.json 2>&1 | tee /tmp/muon_3090_default/run.log | tail -20
RC1=${PIPESTATUS[0]}
echo "Scenario 1 exit code: $RC1"
if [ $RC1 -ne 0 ]; then
    echo "FAIL: 3090 default path crashed — Fix B did not solve it."
    grep -E "device kernel image is invalid|backend='inductor'|Error" /tmp/muon_3090_default/run.log | head -20
    exit 1
fi
echo "PASS: 3090 default (eager Gram) ran cleanly."

# ---- Scenario 2: muon_compile_gram=true (allowed to fail; archives root cause) ----
echo ""
echo "=== Scenario 2: muon_compile_gram=true — ALLOWED TO FAIL ==="
rm -rf /tmp/muon_3090_compile; mkdir -p /tmp/muon_3090_compile
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
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
cfg['optimizer']['muon_compile_gram'] = True   # the Inductor path
json.dump(cfg, open('/tmp/muon_3090_compile/cfg.json', 'w'), indent=2)
print('Config written for Scenario 2 (Inductor opt-in).')
"

python main.py train /tmp/muon_3090_compile/cfg.json 2>&1 | tee /tmp/muon_3090_compile/run.log | tail -30
RC2=${PIPESTATUS[0]}
echo "Scenario 2 exit code: $RC2  (non-zero is acceptable — recording root cause)"
if [ $RC2 -ne 0 ]; then
    echo "Scenario 2 evidence:"
    grep -E "device kernel image is invalid|backend='inductor'|Triton Error|RuntimeError" /tmp/muon_3090_compile/run.log | head -10
fi

echo ""
echo "============================================================"
echo " 3090 smoke summary: Scenario 1 RC=$RC1, Scenario 2 RC=$RC2"
echo "============================================================"
exit 0
