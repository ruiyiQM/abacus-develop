#!/usr/bin/env bash
#SBATCH --job-name=abacus-gcc-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --chdir=/cluster/home/zhourui/abacus-develop
#SBATCH --output=/cluster/home/zhourui/abacus-develop/build-gcc-openmpi/logs/abacus-gcc-build-%j.out
#SBATCH --error=/cluster/home/zhourui/abacus-develop/build-gcc-openmpi/logs/abacus-gcc-build-%j.err
#SBATCH --export=NONE

set -Eeuo pipefail

: "${HOME:?Slurm did not provide HOME}"
: "${SLURM_JOB_USER:?Slurm did not provide SLURM_JOB_USER}"
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# shellcheck source=/dev/null
source /etc/environ.d/lmod.sh
export MODULEPATH=/cluster/software/lmods

echo "Slurm job: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Allocated build CPUs: ${SLURM_CPUS_PER_TASK}"

/cluster/home/zhourui/abacus-develop/euler/build_abacus_gcc_openmpi.sh
