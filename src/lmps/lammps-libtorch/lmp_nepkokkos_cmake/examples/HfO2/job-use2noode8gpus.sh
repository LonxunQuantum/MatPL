#!/bin/sh
#SBATCH -p 4090
#SBATCH -J kk2n
#SBATCH -N 2
#SBATCH -o out
#SBATCH --exclusive

echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"

module purge

module load cuda/11.6-share openmpi/4.1.6 cmake/3.31.6
source /opt/rh/devtoolset-8/enable #gcc


export PATH=/data/home/wuxingxing/codespace/suzhou/lmpversions/lammps-23-4-opt/build-cu116:$PATH

export OMPI_MCA_btl_openib_allow_ib=1
export OMP_NUM_THREADS=1 

# use 2node with 8 gpus
mpirun -np 8 --map-by ppr:4:node lmp -k on g 4 -sf kk -pk kokkos -in kkin.lmp

# use 1node with 4gpus
# mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos -in kkin.lmp

echo "Job $SLURM_JOB_ID done at " `date`
