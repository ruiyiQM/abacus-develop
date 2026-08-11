#!/usr/bin/env bash

export ABACUS_ROOT="/cluster/home/zhourui/abacus-develop"
export ABACUS_BUILD_ROOT="${ABACUS_ROOT}/build-gcc-openmpi"
export PATH=/cluster/apps/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if ! command -v module >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source /etc/environ.d/lmod.sh
fi
export MODULEPATH="${MODULEPATH:-/cluster/software/lmods}"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load openblas/0.3.24
module load netlib-scalapack/2.2.0
module load cmake/3.27.7

export LD_RUN_PATH="${LD_RUN_PATH:-}"

# shellcheck source=/dev/null
source "${ABACUS_BUILD_ROOT}/toolchain/install/setup"

export PATH="${ABACUS_BUILD_ROOT}/install/bin:${PATH}"

# When the parent batch job uses --export=NONE, allow subsequent srun steps to
# receive the clean module/toolchain environment assembled above.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export SLURM_EXPORT_ENV=ALL
fi

# Safe defaults for large MPI jobs. Increase OMP_NUM_THREADS explicitly for a
# hybrid MPI/OpenMP layout; keep OpenBLAS single-threaded to avoid oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
