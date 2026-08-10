#!/usr/bin/env bash
#SBATCH --job-name=abacus-fde-cuda
#SBATCH --partition=gpu.4h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --export=NONE

set -Eeuo pipefail

readonly ABACUS_ROOT="${ABACUS_ROOT:-/cluster/home/zhourui/abacus-develop}"
export ABACUS_ROOT
export ABACUS_BUILD_JOBS="${SLURM_CPUS_PER_TASK}"
exec bash "${ABACUS_ROOT}/euler/build_abacus_gcc_openmpi_cuda.sh"
