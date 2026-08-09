#!/usr/bin/env bash
#SBATCH --job-name=abacus-mpi-smoke
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --chdir=/cluster/home/zhourui/abacus-develop
#SBATCH --output=/cluster/home/zhourui/abacus-develop/build-gcc-openmpi/logs/mpi-smoke-%j.out
#SBATCH --error=/cluster/home/zhourui/abacus-develop/build-gcc-openmpi/logs/mpi-smoke-%j.err
#SBATCH --export=NONE

set -Eeuo pipefail

export PATH=/cluster/apps/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# shellcheck source=/dev/null
source /cluster/home/zhourui/abacus-develop/euler/abacus_gcc_openmpi_env.sh

srun --mpi=pmi2 --cpu-bind=cores --label /bin/hostname
srun --mpi=pmi2 --cpu-bind=cores --label \
    /cluster/home/zhourui/abacus-develop/build-gcc-openmpi/install/bin/abacus --version
