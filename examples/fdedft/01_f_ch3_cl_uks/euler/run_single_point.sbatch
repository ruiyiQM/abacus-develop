#!/usr/bin/env bash
#SBATCH --job-name=fde-example
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1800M
#SBATCH --hint=nomultithread
#SBATCH --export=NONE
#SBATCH --output=fde-example-%j.out

set -Eeuo pipefail

readonly SOURCE_ROOT=${1:-/cluster/home/zhourui/abacus-develop}
readonly RUN_ROOT=${2:-/cluster/scratch/${SLURM_JOB_USER:?}/fdedft-example-${SLURM_JOB_ID:?}}
readonly CASE_SOURCE=${SOURCE_ROOT}/examples/fdedft/01_f_ch3_cl_uks
readonly RUNTIME_ENV=${SOURCE_ROOT}/euler/abacus_gcc_openmpi_env.sh
readonly ABACUS_BIN=${SOURCE_ROOT}/build-gcc-openmpi/install/bin/abacus
readonly WORKFLOW_TOOL=${SOURCE_ROOT}/tools/fde/fde_workflow.py
readonly CONFIGURE_TOOL=${SOURCE_ROOT}/tools/fde/euler/configure_slurm_workflow.py

for path in "${CASE_SOURCE}" "${RUNTIME_ENV}" "${ABACUS_BIN}" "${WORKFLOW_TOOL}" "${CONFIGURE_TOOL}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path is missing: ${path}" >&2
        exit 2
    fi
done
if [[ ! -d "${CASE_SOURCE}/resources" ]]; then
    echo "ERROR: fetch the pinned example resources on the login node before submitting" >&2
    echo "  cd ${CASE_SOURCE} && python3 fetch_default_resources.py" >&2
    exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
    echo "ERROR: run directory already exists: ${RUN_ROOT}" >&2
    exit 3
fi

mkdir -p "${RUN_ROOT}"
cp -a "${CASE_SOURCE}/." "${RUN_ROOT}/"

# shellcheck source=/dev/null
source "${RUNTIME_ENV}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL

cd "${RUN_ROOT}"
python3 prepare_example.py --abacus "${ABACUS_BIN}" --mpi-ranks 1
python3 "${CONFIGURE_TOOL}" workflow.json \
    --output workflow.json \
    --abacus "${ABACUS_BIN}" \
    --ranks "${SLURM_NTASKS}" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --srun /cluster/apps/slurm/bin/srun \
    --work-directory "${RUN_ROOT}/work" \
    --force
python3 "${WORKFLOW_TOOL}" validate workflow.json
/usr/bin/time -v python3 "${WORKFLOW_TOOL}" run workflow.json | tee fdedft.output

test -s work/fde_pes.tsv
test -s work/uks_two_fermi/postprocess/fde_diabatic.fde_diabatic.tsv
echo "SUCCESS: ${RUN_ROOT}/work/fde_pes.tsv"
