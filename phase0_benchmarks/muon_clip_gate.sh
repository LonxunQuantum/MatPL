#!/bin/bash
#SBATCH --job-name=muon_clip_gate
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/muon_clip_gate.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

python - << 'PY'
import sys
from src.user.optimizer_param import OptimizerParam

def case(label, opt_block, expect_pass):
    try:
        op = OptimizerParam()
        op.set_optimizer({"optimizer": opt_block})
        ok = expect_pass
        msg = "no exception"
    except Exception as e:
        ok = (not expect_pass)
        msg = f"raised: {str(e)[:120]}"
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label} -- {msg}")
    return ok

print("=== MUON gradient-clipping gate tests ===")
results = []
results.append(case(
    "MUON without max_norm/clip_value -> ERROR",
    {"optimizer": "MUON"},
    expect_pass=False,
))
results.append(case(
    "MUON + max_norm=2.0 -> pass",
    {"optimizer": "MUON", "max_norm": 2.0, "norm_type": 2},
    expect_pass=True,
))
results.append(case(
    "MUON + clip_value=0.5 -> pass",
    {"optimizer": "MUON", "clip_value": 0.5},
    expect_pass=True,
))
results.append(case(
    "ADAM no clipping (no regression) -> pass",
    {"optimizer": "ADAM"},
    expect_pass=True,
))
results.append(case(
    "ADAMW no clipping (no regression) -> pass",
    {"optimizer": "ADAMW"},
    expect_pass=True,
))
results.append(case(
    "LKF no clipping (no regression) -> pass",
    {"optimizer": "LKF"},
    expect_pass=True,
))
results.append(case(
    "GKF no clipping (no regression) -> pass",
    {"optimizer": "GKF"},
    expect_pass=True,
))

print()
if all(results):
    print(f"All {len(results)} gate tests PASSED")
    sys.exit(0)
else:
    fails = sum(1 for r in results if not r)
    print(f"FAIL: {fails}/{len(results)} cases failed")
    sys.exit(1)
PY
RC=$?
echo "exit code: $RC"
[ $RC -eq 0 ] && echo "PASS: gate validation" || { echo "FAIL"; exit 1; }
