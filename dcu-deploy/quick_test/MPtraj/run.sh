#!/bin/bash -l
#SBATCH --job-name=matpl_test
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=dcu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -eo pipefail

module purge

source /public/home/pwmat/wuxing/MatPL-main-2026.3/dcu-deploy/scnet/setup-dcu-env.sh
source /public/home/pwmat/wuxing/MatPL-main-2026.3/env.sh

which MATPL
MATPL train nep.json
