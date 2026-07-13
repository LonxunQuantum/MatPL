#!/bin/sh
#SBATCH -p 3080ti,new3080ti,q3
#SBATCH -J 1n1g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1

echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"

export CUDA_VISIBLE_DEVICES='0'
module purge

module load cuda/11.6-share openmpi/4.1.6 cmake/3.31.6
source /opt/rh/devtoolset-8/enable #gcc

export PATH=/data/home/wuxingxing/xcode/lammps-qnep-pppm/build:$PATH

#export PATH=/data/home/wuxingxing/codespace/suzhou/lmpversions/lammps-maindb64-cvatom/build-32:$PATH
#export OMPI_MCA_btl_openib_allow_ib=1
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
export OMP_NUM_THREADS=1  # 根据您的任务设置

#mpirun -np 4 lmp -in lmp.incpu

mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos -in kkin.lmp

#nsys profile --trace=cuda,cublas,osrt --output=c3b256-4g-lammps.nsys-rep --force-overwrite true  mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos -in kkin.lmp


echo "Job $SLURM_JOB_ID done at " `date`
