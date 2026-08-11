#!/usr/bin/env bash
#SBATCH --job-name=fde-workflow
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1800M
#SBATCH --hint=nomultithread
#SBATCH --requeue
#SBATCH --export=NONE
#SBATCH --output=fde-workflow-%A_%a.out

set -Eeuo pipefail

input=${1:?usage: run_fde_workflow.sbatch SPEC_OR_MANIFEST [WORKFLOW_TOOL] [RUNTIME_ENV] [SPEC_COLUMN] [WORKFLOW_SHA256]}
source_root=/cluster/home/${SLURM_JOB_USER:?}/abacus-develop
workflow_tool=${2:-${source_root}/tools/fde/fde_workflow.py}
runtime_env=${3:-${source_root}/euler/abacus_gcc_openmpi_env.sh}
spec_column=${4:-auto}
expected_workflow_sha256=${5:-}

export PATH=/cluster/apps/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONDONTWRITEBYTECODE=1

input=$(realpath -e "${input}")
workflow_tool=$(realpath -e "${workflow_tool}")
runtime_env=$(realpath -e "${runtime_env}")

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_id=${SLURM_ARRAY_TASK_ID}
    spec=$(awk -F '\t' -v task_id="${task_id}" -v requested="${spec_column}" '
        NR == 1 {
            task_column = 0
            spec_column = 0
            for (column = 1; column <= NF; ++column) {
                if ($column == "task_id") task_column = column
                if (requested != "auto" && $column == requested) spec_column = column
                if (requested == "auto" && ($column == "spec" || $column == "spec_path")) {
                    if (spec_column != 0) exit 4
                    spec_column = column
                }
            }
            if (task_column == 0 || spec_column == 0) exit 3
            next
        }
        $task_column == task_id {print $spec_column; matches += 1}
        END {if (matches != 1) exit 5}
    ' "${input}") || {
        echo "ERROR: could not select task ${task_id} from ${input}" >&2
        exit 2
    }
else
    task_id=single
    spec=${input}
fi

spec=$(realpath -e "${spec}")
if [[ -n "${expected_workflow_sha256}" ]]; then
    actual_workflow_sha256=$(sha256sum "${workflow_tool}" | awk '{print $1}')
    if [[ "${actual_workflow_sha256}" != "${expected_workflow_sha256}" ]]; then
        echo "ERROR: workflow tool SHA-256 differs from the submitted value" >&2
        exit 3
    fi
fi

# shellcheck source=/dev/null
source "${runtime_env}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL

echo "start=$(date --iso-8601=seconds)"
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${task_id}"
echo "spec=${spec}"
echo "spec_sha256=$(sha256sum "${spec}" | awk '{print $1}')"
echo "workflow=${workflow_tool}"
echo "workflow_sha256=$(sha256sum "${workflow_tool}" | awk '{print $1}')"
echo "layout=${SLURM_NTASKS}x${SLURM_CPUS_PER_TASK} cores=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"

python3 "${workflow_tool}" validate "${spec}"
/usr/bin/time -v python3 "${workflow_tool}" run "${spec}"

echo "finish=$(date --iso-8601=seconds)"
