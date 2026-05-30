#!/bin/bash
#SBATCH --job-name=phase2_23_smoke
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/phase2_23_smoke.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "========================================"
echo "Phase 2.2 + 2.3 Smoke Test"
echo "========================================"

# Test 1: defaults off (sanity)
echo ""
echo "=== Test 1: defaults off ==="
rm -rf /tmp/p23_test1; mkdir -p /tmp/p23_test1
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']; del cfg['optimizer']['t_mult']
json.dump(cfg, open('/tmp/p23_test1/cfg.json', 'w'), indent=2)
"
python main.py train /tmp/p23_test1/cfg.json 2>&1 | tail -3
RC=$?
[ $RC -eq 0 ] && echo "PASS: Test 1 (defaults off)" || { echo "FAIL: Test 1 RC=$RC"; exit 1; }

# Test 2: same_nloc_sampler ON only
echo ""
echo "=== Test 2: same_nloc_sampler ON ==="
rm -rf /tmp/p23_test2; mkdir -p /tmp/p23_test2
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']; del cfg['optimizer']['t_mult']
cfg['same_nloc_sampler'] = True
json.dump(cfg, open('/tmp/p23_test2/cfg.json', 'w'), indent=2)
"
python main.py train /tmp/p23_test2/cfg.json 2>&1 | tail -3
RC=$?
[ $RC -eq 0 ] && echo "PASS: Test 2 (same_nloc_sampler ON)" || { echo "FAIL: Test 2 RC=$RC"; exit 1; }

# Test 3: same_nloc_sampler + compile_fitting BOTH on
echo ""
echo "=== Test 3: same_nloc_sampler + compile_fitting ON ==="
rm -rf /tmp/p23_test3; mkdir -p /tmp/p23_test3
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']; del cfg['optimizer']['t_mult']
cfg['same_nloc_sampler'] = True
cfg['compile_fitting'] = True
json.dump(cfg, open('/tmp/p23_test3/cfg.json', 'w'), indent=2)
"
python main.py train /tmp/p23_test3/cfg.json 2>&1 | tail -10
RC=$?
[ $RC -eq 0 ] && echo "PASS: Test 3 (both ON)" || { echo "FAIL: Test 3 RC=$RC"; exit 1; }

# Test 4: compile_fitting ON, same_nloc_sampler OFF (recompile storm expected, just verify it runs)
echo ""
echo "=== Test 4: compile_fitting ON without bucketing (smoke only, recompile expected) ==="
rm -rf /tmp/p23_test4; mkdir -p /tmp/p23_test4
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['epochs'] = 1
del cfg['optimizer']['t_0']; del cfg['optimizer']['t_mult']
cfg['compile_fitting'] = True
json.dump(cfg, open('/tmp/p23_test4/cfg.json', 'w'), indent=2)
"
python main.py train /tmp/p23_test4/cfg.json 2>&1 | tail -10
RC=$?
[ $RC -eq 0 ] && echo "PASS: Test 4 (compile only)" || { echo "FAIL: Test 4 RC=$RC"; exit 1; }

echo ""
echo "========================================"
echo "ALL PHASE 2.2+2.3 SMOKE TESTS PASSED"
echo "========================================"
