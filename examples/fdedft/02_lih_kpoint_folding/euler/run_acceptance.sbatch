#!/usr/bin/env bash
#SBATCH --job-name=fde-lih-fold
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=nomultithread
#SBATCH --export=NONE
#SBATCH --output=fde-lih-fold-%j.out

set -Eeuo pipefail

readonly SOURCE_ROOT=${1:-/cluster/home/${SLURM_JOB_USER:?}/abacus-develop}
readonly RUN_ROOT=${2:-/cluster/scratch/${SLURM_JOB_USER}/abacus_test/fde-lih-fold-${SLURM_JOB_ID:?}}
readonly ABACUS_BIN=${3:-${ABACUS_FDE_BINARY:-${SOURCE_ROOT}/build-gcc-openmpi/install/bin/abacus}}
readonly RUNTIME_ENV=${4:-${ABACUS_FDE_RUNTIME_ENV:-${SOURCE_ROOT}/euler/abacus_gcc_openmpi_env.sh}}
readonly CASE_SOURCE=${SOURCE_ROOT}/examples/fdedft/02_lih_kpoint_folding
readonly RESOURCE_SOURCE=${5:-${ABACUS_FDE_RESOURCE_ROOT:-${CASE_SOURCE}/resources}}
readonly COMPARE_TOOL=${SOURCE_ROOT}/tools/fde/compare_band_folding.py

for path in "${CASE_SOURCE}" "${RESOURCE_SOURCE}" "${ABACUS_BIN}" "${RUNTIME_ENV}" "${COMPARE_TOOL}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path is missing: ${path}" >&2
        exit 2
    fi
done
if [[ -e "${RUN_ROOT}" ]]; then
    echo "ERROR: run directory already exists: ${RUN_ROOT}" >&2
    exit 3
fi

mkdir -p "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/primitive" "${RUN_ROOT}/supercell" "${RUN_ROOT}/resources"
for filename in README.md default_resources.json fetch_default_resources.py prepare_example.py; do
    cp "${CASE_SOURCE}/${filename}" "${RUN_ROOT}/${filename}"
done
cp -a "${CASE_SOURCE}/primitive/." "${RUN_ROOT}/primitive/"
cp -a "${CASE_SOURCE}/supercell/." "${RUN_ROOT}/supercell/"
cp -a "${RESOURCE_SOURCE}/." "${RUN_ROOT}/resources/"

# shellcheck source=/dev/null
source "${RUNTIME_ENV}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL

cd "${RUN_ROOT}"
python3 prepare_example.py
for system in primitive supercell; do
    (
        cd "${system}"
        srun --mpi=pmi2 --nodes=1 --ntasks="${SLURM_NTASKS}" \
            --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
            "${ABACUS_BIN}" > abacus.log 2>&1
        test -s result.fde_density
        test -s result.fde_kbands
    )
done
python3 "${COMPARE_TOOL}" \
    --primitive "${RUN_ROOT}/primitive/result.fde_kbands" \
    --supercell "${RUN_ROOT}/supercell/result.fde_kbands" \
    --tolerance-ry 1e-6 \
    --output "${RUN_ROOT}/band_folding_result.json"
echo "SUCCESS: ${RUN_ROOT}/band_folding_result.json"
