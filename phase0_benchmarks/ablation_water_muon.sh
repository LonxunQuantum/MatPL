#!/bin/bash
#SBATCH --job-name=abl_w_muon
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/ablation_water_muon.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/water_5k_base.json"
export PYTHONPATH="$CODE_DIR"

echo "=== Ablation (water_5k, 2 elements): HybridMuon (10 epochs, seed=4972) ==="
WORK=/tmp/abl_water_muon
rm -rf $WORK; mkdir -p $WORK
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['optimizer'] = 'MUON'
cfg['optimizer']['epochs'] = 10
cfg['optimizer']['print_freq'] = 200
cfg['optimizer']['learning_rate'] = 1e-3
cfg['optimizer']['max_norm'] = 2.0
cfg['optimizer']['norm_type'] = 2
cfg['optimizer']['muon_mode'] = 'slice'
cfg['optimizer']['muon_enable_gram'] = True
cfg['optimizer']['muon_flash'] = False
cfg['optimizer']['muon_magma'] = False
cfg['optimizer']['muon_lr_adjust'] = 0.0
cfg['optimizer']['muon_lr_adjust_coeff'] = 0.18
cfg['optimizer']['momentum'] = 0.95
cfg['optimizer']['weight_decay'] = 1e-3
json.dump(cfg, open('$WORK/cfg.json', 'w'), indent=2)
print('MUON config ready')
"
cd $WORK
python $CODE_DIR/main.py train $WORK/cfg.json 2>&1
RC=$?
cd $CODE_DIR
echo "exit code: $RC"
[ $RC -eq 0 ] && echo "PASS: MUON water ablation finished" || { echo "FAIL"; exit 1; }
