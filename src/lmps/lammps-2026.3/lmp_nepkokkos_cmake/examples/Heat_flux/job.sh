#!/bin/sh
#SBATCH -p q4
#SBATCH -J heat_flux
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -o out

echo "Starting job $SLURM_JOB_ID at " `date`
echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"

module purge
module load cuda/11.6-share openmpi/4.1.6 gcc/8.3.1
export OMP_NUM_THREADS=1
export PATH=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/lammps-stable_22Jul2025_update4/AMPERE86:$PATH

mpirun -np $SLURM_NPROCS --bind-to numa --map-by ppr:$SLURM_NPROCS:node lmp -k on g $SLURM_NPROCS -sf kk -pk kokkos -in graphene.in > lmp.out 2>&1

echo "Job $SLURM_JOB_ID done at " `date`
