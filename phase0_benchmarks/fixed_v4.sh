#!/bin/bash
#SBATCH -p 4090
#SBATCH -J MatPL
#SBATCH -N 1
#SBATCH -o fixed_v4.log
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

module purge
module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate
set -e
export PYTHONPATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL:$PYTHONPATH
export PATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/src/bin:$PATH
MATPL_ROOT=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL
BENCH_DIR=$MATPL_ROOT/phase0_benchmarks

# 删老 model_record，跑 mixed 三档
rm -rf model_record
timeout 120 python ../main.py train small_Si_mixed.json 2>&1 | tee train_small_Si_mixed_v4.log
rm -rf model_record
timeout 120 python ../main.py train medium_C_mixed.json 2>&1 | tee train_medium_C_mixed_v4.log
rm -rf model_record
timeout 300 python ../main.py train large_water_mixed.json 2>&1 | tee train_large_water_mixed_v4.log

# 三方对比
for tag in small_Si medium_C large_water; do
  echo "=== $tag ==="
  echo -n "A) fp64 CalcOps:     "
  grep "^Epoch:" train_${tag}.log 2>/dev/null | tail -50 | awk '{for(i=1;i<=NF;i++) if($i=="Time") {t+=$(i+1);n++}} END{printf "%.4f s/step\n", t/n}'
  echo -n "B) fp32 CalcOps 2b only: "
  grep "^Epoch:" train_${tag}_mixed_v4.log 2>/dev/null | tail -50 | awk '{for(i=1;i<=NF;i++) if($i=="Time") {t+=$(i+1);n++}} END{printf "%.4f s/step\n", t/n}'
done

