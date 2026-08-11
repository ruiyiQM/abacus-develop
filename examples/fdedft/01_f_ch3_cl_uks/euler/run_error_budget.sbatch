#!/usr/bin/env bash
#SBATCH --job-name=fde-budget
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1800M
#SBATCH --hint=nomultithread
#SBATCH --array=0-2%3
#SBATCH --export=NONE
#SBATCH --output=fde-budget-%A_%a.out

set -Eeuo pipefail

readonly SOURCE_ROOT=${1:-/cluster/home/zhourui/abacus-develop}
readonly BUDGET_ROOT=${2:-/cluster/scratch/${SLURM_JOB_USER:?}/abacus_test/fde-scientific-budget-${SLURM_ARRAY_JOB_ID:?}}
readonly CASE_SOURCE=${SOURCE_ROOT}/examples/fdedft/01_f_ch3_cl_uks
readonly ABACUS_BIN=${ABACUS_FDE_BINARY:-${SOURCE_ROOT}/build-gcc-openmpi/install/bin/abacus}
readonly RUNTIME_ENV=${ABACUS_FDE_RUNTIME_ENV:-${SOURCE_ROOT}/euler/abacus_gcc_openmpi_env.sh}
readonly WORKFLOW_TOOL=${SOURCE_ROOT}/tools/fde/fde_workflow.py
readonly CONFIGURE_TOOL=${SOURCE_ROOT}/tools/fde/euler/configure_slurm_workflow.py
readonly COMPARE_TOOL=${SOURCE_ROOT}/tools/fde/compare_scientific_runs.py

case ${SLURM_ARRAY_TASK_ID:?} in
    0)
        variant=production
        cutoff=40
        scf_tolerance=3e-6
        freeze_thaw_tolerance=1e-5
        energy_tolerance=2e-5
        maximum_scf_iterations=200
        ;;
    1)
        variant=tight_scf
        cutoff=40
        scf_tolerance=1e-6
        freeze_thaw_tolerance=3e-6
        energy_tolerance=5e-6
        maximum_scf_iterations=300
        ;;
    2)
        variant=cutoff60
        cutoff=60
        scf_tolerance=1e-6
        freeze_thaw_tolerance=3e-6
        energy_tolerance=5e-6
        maximum_scf_iterations=300
        ;;
    *)
        echo "ERROR: unsupported benchmark task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

readonly RUN_ROOT=${BUDGET_ROOT}/${variant}
if [[ -e "${RUN_ROOT}" ]]; then
    echo "ERROR: benchmark run already exists: ${RUN_ROOT}" >&2
    exit 3
fi
if [[ ! -d "${CASE_SOURCE}/resources" ]]; then
    echo "ERROR: fetch pinned resources in ${CASE_SOURCE} before submission" >&2
    exit 4
fi
if [[ ! -x "${ABACUS_BIN}" ]]; then
    echo "ERROR: ABACUS executable is missing: ${ABACUS_BIN}" >&2
    exit 5
fi
if [[ ! -r "${RUNTIME_ENV}" ]]; then
    echo "ERROR: runtime environment is missing: ${RUNTIME_ENV}" >&2
    exit 5
fi

mkdir -p "${RUN_ROOT}"
for filename in INPUT KPT STRU default_resources.json geometry.csv prepare_example.py workflow.template.json scientific_error_budget.json; do
    cp "${CASE_SOURCE}/${filename}" "${RUN_ROOT}/${filename}"
done

# shellcheck source=/dev/null
source "${RUNTIME_ENV}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL

cd "${RUN_ROOT}"
python3 prepare_example.py \
    --abacus "${ABACUS_BIN}" \
    --mpi-ranks 1 \
    --pseudo-dir "${CASE_SOURCE}/resources/pseudopotentials" \
    --orbital-dir "${CASE_SOURCE}/resources/orbitals" \
    --variant-label "${variant}" \
    --ecutwfc "${cutoff}" \
    --scf-density-tolerance "${scf_tolerance}" \
    --freeze-thaw-density-tolerance "${freeze_thaw_tolerance}" \
    --energy-tolerance-ry "${energy_tolerance}" \
    --maximum-scf-iterations "${maximum_scf_iterations}"

readonly SPEC=${RUN_ROOT}/workflow-${variant}.json
python3 "${CONFIGURE_TOOL}" "${SPEC}" \
    --output "${SPEC}" \
    --abacus "${ABACUS_BIN}" \
    --ranks "${SLURM_NTASKS}" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --srun /cluster/apps/slurm/bin/srun \
    --work-directory "${RUN_ROOT}/work-${variant}" \
    --force
python3 "${WORKFLOW_TOOL}" validate "${SPEC}"
/usr/bin/time -v python3 "${WORKFLOW_TOOL}" run "${SPEC}" \
    | tee "fdedft-${variant}.output"

set +e
python3 "${COMPARE_TOOL}" \
    scientific_error_budget.json \
    "${CASE_SOURCE}/reference/fde_pes.tsv.ref" \
    "${RUN_ROOT}/work-${variant}/fde_pes.tsv" \
    --candidate-convergence "${RUN_ROOT}/work-${variant}" \
    --output "${RUN_ROOT}/comparison_to_committed_reference.json"
comparison_status=$?
set -e
printf '%s\n' "${comparison_status}" > comparison_exit_status.txt
if [[ "${variant}" != cutoff60 && ${comparison_status} -ne 0 ]]; then
    echo "ERROR: ${variant} exceeds the committed reproducibility budget" >&2
    exit "${comparison_status}"
fi
echo "SUCCESS: ${RUN_ROOT}"
