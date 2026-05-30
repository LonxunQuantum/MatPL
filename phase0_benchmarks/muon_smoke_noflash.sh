#!/bin/bash
#SBATCH --job-name=muon_noflash
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/muon_smoke_noflash.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "=== Phase 3.3 smoke test: HybridMuon on small_Si (NO Triton) ==="
rm -rf /tmp/muon_noflash; mkdir -p /tmp/muon_noflash
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['optimizer'] = 'MUON'
cfg['optimizer']['epochs'] = 2
cfg['optimizer']['print_freq'] = 10
for k in ('t_0', 't_mult', 'norm_type', 'max_norm'):
    cfg['optimizer'].pop(k, None)
cfg['optimizer']['muon_mode'] = 'slice'
cfg['optimizer']['muon_enable_gram'] = True
cfg['optimizer']['muon_flash'] = False
cfg['optimizer']['muon_magma'] = False
cfg['optimizer']['muon_lr_adjust'] = 0.0
cfg['optimizer']['muon_lr_adjust_coeff'] = 0.18
json.dump(cfg, open('/tmp/muon_noflash/cfg.json', 'w'), indent=2)
print('Config written with optimizer=MUON (Triton disabled)')
"

python main.py train /tmp/muon_noflash/cfg.json 2>&1 | tail -30
RC=$?
echo "exit code: $RC"

[ $RC -eq 0 ] && echo "PASS: HybridMuon (no-Triton) smoke test passed" || { echo "FAIL"; exit 1; }
