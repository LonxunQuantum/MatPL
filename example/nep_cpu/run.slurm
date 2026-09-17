#!/bin/bash
#SBATCH --job-name=nep-cpu
#SBATCH --partition=3080ti,new3080ti,q3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=cpu-train-%j.out
set -euo pipefail
source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
conda activate matpl-2026.3

# Set device=cpu in nep.json. No GPU resources are requested or used.
export CUDA_VISIBLE_DEVICES=""
export PYTHONUTF8=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

cd "$SLURM_SUBMIT_DIR"
srun --cpu-bind=cores --kill-on-bad-exit=1 \
    python /data/home/wuxingxing/xcode/MatPL-dcu-dev/main.py train nep.json
