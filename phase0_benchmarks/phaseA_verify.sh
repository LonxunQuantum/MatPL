#!/bin/bash
#SBATCH --job-name=phaseA_verify
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/phaseA_verify.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "=== Phase A Verification ==="
echo "1. ABI + dtype guard unit test"
python -c "
import torch
torch.ops.load_library('src/op/build/lib/libCalcOps_bind.so')
CalcOps = torch.ops.CalcOps_cuda
device = torch.device('cuda:0')
natoms, neigh, n_max, n_base, ntypes = 4, 10, 4, 4, 1
coeff2 = torch.zeros(ntypes, ntypes, n_max+1, n_base+1, dtype=torch.float64, device=device)
d12 = torch.zeros(natoms, neigh, 4, dtype=torch.float64, device=device)
NL = torch.full((natoms, neigh), -1, dtype=torch.int64, device=device)
atom_map = torch.zeros(natoms, dtype=torch.int64, device=device)
feat = torch.zeros(natoms, (n_max+1)*4, dtype=torch.float64, device=device, requires_grad=True)
out = CalcOps.calculateNepFeat(coeff2, d12, NL, atom_map, feat, 6.0, (n_max+1)*4, 0)
print('[PASS] calculateNepFeat fp64 OK, shape:', out[0].shape)

# fp32 should raise
try:
    CalcOps.calculateNepFeat(coeff2.float(), d12.float(), NL, atom_map, feat.float(), 6.0, (n_max+1)*4, 0)
    print('[FAIL] fp32 should have raised!')
except RuntimeError as e:
    if 'float64' in str(e):
        print('[PASS] fp32 dtype guard fires correctly')
    else:
        print('[FAIL] Unexpected error:', e)

# Test force op
force = torch.zeros(natoms, 3, dtype=torch.float64, device=device)
dE = torch.zeros(natoms, neigh, dtype=torch.float64, device=device)
Ri_d = torch.zeros(natoms, neigh, 3, dtype=torch.float64, device=device)
CalcOps.calculateNepForce(NL, dE, Ri_d, force)
print('[PASS] calculateNepForce fp64 OK')

try:
    CalcOps.calculateNepForce(NL, dE.float(), Ri_d.float(), force.float())
    print('[FAIL] force fp32 should have raised!')
except RuntimeError as e:
    if 'float64' in str(e):
        print('[PASS] force fp32 dtype guard fires correctly')
    else:
        print('[FAIL] Unexpected error:', e)
print('Unit tests done.')
"
RC1=$?

echo ""
echo "2. Full training smoke (small_Si fp64, 1 epoch)"
rm -rf /tmp/phaseA_test; mkdir -p /tmp/phaseA_test
python -c "
import json
cfg = json.load(open('phase0_benchmarks/small_Si.json'))
cfg['optimizer']['epochs'] = 1
cfg['optimizer']['print_freq'] = 100
json.dump(cfg, open('/tmp/phaseA_test/cfg.json', 'w'), indent=2)
print('Config written')
"
set -o pipefail
python main.py train /tmp/phaseA_test/cfg.json 2>&1 | tail -10
RC2=${PIPESTATUS[0]}

echo ""
echo "=== Results ==="
echo "Unit test exit: $RC1"
echo "Training exit: $RC2"
if [ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; then
    echo "PHASE A VERIFICATION: PASS"
else
    echo "PHASE A VERIFICATION: FAIL"
    exit 1
fi
