#!/bin/bash
#SBATCH --job-name=omat-mix-dist
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/home/wuxingxing/datas/training_test/mgpus/omat24/cout/mix-distribution-%j.log
set -euo pipefail
source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
conda activate matpl-2026.3
source /opt/rh/devtoolset-8/enable
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OMAT24_SOURCE="${OMAT24_SOURCE:-/data/public/wuxingxing/metadata/decompress/Omat24/train}"
python "${SCRIPT_DIR}/analyze_omat24_mix_batchsize.py" \
    --source "$OMAT24_SOURCE" \
    --output "$SCRIPT_DIR"
