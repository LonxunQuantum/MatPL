#!/bin/bash
#SBATCH --job-name=abl_w_adamw
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/ablation_water_adamw.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/water_5k_base.json"
export PYTHONPATH="$CODE_DIR"

echo "=== Ablation (water_5k, 2 elements): AdamW (10 epochs, seed=4972) ==="
WORK=/tmp/abl_water_adamw
rm -rf $WORK; mkdir -p $WORK
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['optimizer'] = 'ADAMW'
cfg['optimizer']['epochs'] = 10
cfg['optimizer']['print_freq'] = 200
cfg['optimizer']['learning_rate'] = 1e-3
cfg['optimizer']['weight_decay'] = 1e-3
json.dump(cfg, open('$WORK/cfg.json', 'w'), indent=2)
print('ADAMW config ready')
"
cd $WORK
python $CODE_DIR/main.py train $WORK/cfg.json 2>&1
RC=$?
cd $CODE_DIR
echo "exit code: $RC"
[ $RC -eq 0 ] && echo "PASS: ADAMW water ablation finished" || { echo "FAIL"; exit 1; }
