#!/usr/bin/env bash
set -eo pipefail

work=$1
cd -- "$work"
export XDG_CACHE_HOME="$work/.cache"
# Keep multiprocessing's AF_UNIX address short while storing files under work.
# The worker below replaces this shell and retains its PID and working directory.
export TMPDIR="/proc/$$/cwd/.tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p -- "$work/.tmp" "$MPLCONFIGDIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
export MATPL_QA_CODE=$2
requested_gpus=$3
cpus=$4
conda_sh=$5
conda_env=$6
cuda_module=$7

if [[ -n "$cuda_module" ]]; then
    module load "$cuda_module"
fi
source "$conda_sh"
conda activate "$conda_env"
source "$MATPL_QA_CODE/env.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MATPL_DDP_TIMEOUT_SECONDS="${MATPL_DDP_TIMEOUT_SECONDS:-120}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "$requested_gpus" == 0 ]]; then
    export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
fi
printf 'MatPL %s\n' "$(command -v MATPL)"
printf 'Slurm job %s, CPUS %s, GPUs %s\n' "${SLURM_JOB_ID:-local}" "$cpus" "$requested_gpus"
exec python "$8/worker.py" "$work"
