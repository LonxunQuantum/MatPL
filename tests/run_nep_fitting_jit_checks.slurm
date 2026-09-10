#!/bin/bash
#SBATCH --job-name=nep-fit-jit
#SBATCH --partition=3090,q4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=.validation/nep-fitting-jit-%j.log
set -eo pipefail
source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
conda activate matpl-2026.3
module load cuda/11.8-share cmake/3.31.6
source /opt/rh/devtoolset-8/enable
set -u
repo=${SLURM_SUBMIT_DIR}
case_dir=/data/home/wuxingxing/datas/training_test/mgpus/omat24/mini_data_test
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
mkdir -p .validation

python -m pytest tests/test_nep_fitting_jit.py tests/test_nep_fitting_jit_tools.py \
  tests/test_nep_fused_fitting.py tests/test_nep_fused_runtime.py \
  tests/test_nep_fused_training.py \
  src/test/test_nep_electric/test_mb_secondgrad_kernel.py -q

arch_dir=.validation/jit-arches
rm -rf "$arch_dir"
python tests/compile_nep_fitting_jit_arches.py \
  --header src/op/kernel/nep_fitting_jit_source.h --output-dir "$arch_dir" \
  --sm 60 70 86 89
python tests/check_nep_fitting_jit_cubin.py "$arch_dir"/*.cubin \
  --output "$arch_dir/resources.json"

stamp=$(date +%Y%m%d-%H%M%S)
cache=.validation/omat-jit-cache-$stamp
checkpoint="$case_dir/model_record/nep_model.ckpt"
if [[ -f "$checkpoint" ]]; then
  common=(--case "$case_dir" --warmup "${WARMUP:-10}" --steps "${STEPS:-30}" \
          --batch-size "${BATCH_SIZE:-256}" --jit-cache "$cache")
  rm -rf "$cache"
  python -u tests/benchmark_nep_fused_fitting.py "${common[@]}" \
    --output ".validation/nep-fitting-jit-cold-$stamp.json"
  python -u tests/benchmark_nep_fused_fitting.py "${common[@]}" \
    --output ".validation/nep-fitting-jit-hot-$stamp.json"
else
  echo "Skipping OMat24 benchmark: checkpoint not found at $checkpoint"
fi
