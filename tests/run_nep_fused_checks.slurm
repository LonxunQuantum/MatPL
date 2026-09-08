#!/bin/bash
#SBATCH --job-name=nep-fused-checks
#SBATCH --partition=3090,q4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=.validation/fitting-%j.log
set -eo pipefail
source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
conda activate matpl-2026.3
module load cuda/11.8-share cmake/3.31.6
source /opt/rh/devtoolset-8/enable
set -u
repo=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
case_dir=/data/home/wuxingxing/datas/training_test/mgpus/omat24/mini_data_test
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
mkdir -p .validation
python -m unittest discover -s tests -p 'test_nep_fused*.py' -v
stamp=$(date +%Y%m%d-%H%M%S)
args=(--case "$case_dir" --warmup "${WARMUP:-20}" --steps "${STEPS:-100}"
      --batch-size "${BATCH_SIZE:-256}" --output ".validation/fitting-$stamp.json")
if [[ -n ${MAX_BATCH:-} ]]; then args+=(--max-batch "$MAX_BATCH"); fi
python -u tests/benchmark_nep_fused_fitting.py "${args[@]}"
