#!/bin/sh
#SBATCH -p 4090
#SBATCH -J kk2n
#SBATCH -N 2
#SBATCH -o out
#SBATCH --exclusive

echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"

module purge
module load lammps4matpl/2026.3

export OMPI_MCA_btl_openib_allow_ib=1
export OMP_NUM_THREADS=1 

# use 2node with 8 gpus
mpirun -np $SLURM_NPORCS --bind-to numa --map-by ppr:4:node lmp -k on g 4 -sf kk -pk kokkos -in kkin.lmp

# use 1node with 4gpus
# mpirun -np 4 --bind-to numa --map-by ppr:4:node lmp -k on g 4 -sf kk -pk kokkos -in kkin.lmp

echo "Job $SLURM_JOB_ID done at " `date`
