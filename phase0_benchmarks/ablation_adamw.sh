#!/bin/bash
#SBATCH --job-name=muon_abl_adamw
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/ablation_adamw.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
BASE_CFG="$CODE_DIR/phase0_benchmarks/small_Si.json"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "=== Ablation: AdamW (20 epochs, seed=4972) ==="
WORK=/tmp/abl_adamw
rm -rf $WORK; mkdir -p $WORK
python -c "
import json
cfg = json.load(open('$BASE_CFG'))
cfg['optimizer']['optimizer'] = 'ADAMW'
cfg['optimizer']['epochs'] = 20
cfg['optimizer']['print_freq'] = 100
cfg['optimizer']['learning_rate'] = 1e-3
cfg['optimizer']['weight_decay'] = 1e-3
# Drop cosine restarts & grad clipping to match Muon run (Muon doesn't use them)
for k in ('t_0', 't_mult', 'norm_type', 'max_norm'):
    cfg['optimizer'].pop(k, None)
json.dump(cfg, open('$WORK/cfg.json', 'w'), indent=2)
print('ADAMW config ready')
"
cd $WORK
python $CODE_DIR/main.py train $WORK/cfg.json 2>&1
RC=$?
cd $CODE_DIR
echo "exit code: $RC"
[ $RC -eq 0 ] && echo "PASS: ADAMW ablation finished" || { echo "FAIL"; exit 1; }
