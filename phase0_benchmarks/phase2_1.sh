#!/bin/bash
#SBATCH -p 4090
#SBATCH -J MatPL_p21
#SBATCH -N 1
#SBATCH -o phase2_1.log
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

module purge
module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate
set -e
export PYTHONPATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL:$PYTHONPATH
export PATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/src/bin:$PATH

cd /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks

run_one () {
  local cfg="$1"        # e.g. small_Si.json
  local tag="$2"        # log tag
  local mode="$3"       # loop | batched
  local t_limit="$4"    # seconds

  if [ "$mode" = "batched" ]; then
    export MATPL_BATCHED_FITTING=1
  else
    unset MATPL_BATCHED_FITTING
  fi

  rm -rf model_record
  timeout "$t_limit" python ../main.py train "$cfg" 2>&1 \
    | tee "train_${tag}_${mode}_p21.log"
}

run_one small_Si.json    small_Si    loop    180
run_one small_Si.json    small_Si    batched 180
run_one medium_C.json    medium_C    loop    180
run_one medium_C.json    medium_C    batched 180
run_one large_water.json large_water loop    360
run_one large_water.json large_water batched 360

echo
echo "===== PHASE 2.1: loop (default) vs batched (MATPL_BATCHED_FITTING=1) ====="
for tag in small_Si medium_C large_water; do
  echo "=== $tag ==="
  for mode in loop batched; do
    log="train_${tag}_${mode}_p21.log"
    echo -n "$mode: "
    if [ -f "$log" ]; then
      grep "^Epoch:" "$log" 2>/dev/null | tail -50 | awk '
        { for(i=1;i<=NF;i++) if($i=="Time") {t+=$(i+1);n++} }
        END { if(n>0) printf "%.4f s/step over last %d steps  | ", t/n, n; else printf "no Epoch lines  | "; }'
      grep -E "(Etot_RMSE|Force_RMSE|Etot RMSE|Force RMSE|RMSE_)" "$log" 2>/dev/null | tail -1
      echo
    else
      echo "(no log)"
    fi
  done
done
