#!/usr/bin/env bash

set -Eeuo pipefail

source_root=${1:-/cluster/home/${USER:?}/abacus-develop}
scratch=${2:-/cluster/scratch/${USER}/abacus-fde-mpi-scaling}
template=${3:-${ABACUS_FDE_SCALING_TEMPLATE:-}}
binary=${4:-${ABACUS_FDE_BINARY:-${source_root}/build-gcc-openmpi/install/bin/abacus}}
runtime_env=${5:-${ABACUS_FDE_RUNTIME_ENV:-${source_root}/euler/abacus_gcc_openmpi_env.sh}}
source_commit=${6:-}
if [[ -z "${template}" ]]; then
    echo "usage: $0 [SOURCE_ROOT] [SCRATCH_ROOT] TEMPLATE [ABACUS] [RUNTIME_ENV] [SOURCE_COMMIT]" >&2
    exit 2
fi
for path in "${source_root}" "${template}" "${binary}" "${runtime_env}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required scaling path is missing: ${path}" >&2
        exit 2
    fi
done
source_root=$(realpath -e "${source_root}")
template=$(realpath -e "${template}")
binary=$(realpath -e "${binary}")
runtime_env=$(realpath -e "${runtime_env}")
if [[ -z "${source_commit}" ]]; then
    if source_commit=$(git -C "${source_root}" rev-parse HEAD 2>/dev/null); then
        :
    else
        source_commit=unknown
    fi
fi
runner=${source_root}/tools/fde/euler/run_fde_mpi_scaling.sbatch
summarizer=${source_root}/tools/fde/euler/summarize_fde_mpi_scaling.py
for path in "${runner}" "${summarizer}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required scaling tool is missing: ${path}" >&2
        exit 2
    fi
done
if [[ -e "${scratch}/jobs.txt" ]]; then
    echo "ERROR: scaling root already contains a submission: ${scratch}" >&2
    exit 3
fi

mkdir -p "${scratch}/logs" "${scratch}/results"

submit()
{
    local layout=$1
    local omp_threads=$2
    shift 2
    sbatch --parsable --partition=normal.4h --hint=nomultithread \
        --mem-per-cpu=1800M \
        --output="${scratch}/logs/${layout}-%j.out" \
        --error="${scratch}/logs/${layout}-%j.err" \
        "$@" "${runner}" "${layout}" "${omp_threads}" \
        "${source_root}" "${scratch}" "${template}" "${binary}" \
        "${runtime_env}" "${source_commit}"
}

job_1x1=$(submit 1rank_1thread 1 --nodes=1 --ntasks=1 --cpus-per-task=80)
job_4x1=$(submit 4rank_1thread 1 --nodes=1 --ntasks=4 --cpus-per-task=20)
job_20x4=$(submit 20rank_4thread 4 --nodes=1 --ntasks=20 --cpus-per-task=4)
job_40x4=$(submit 40rank_4thread 4 --nodes=2 --ntasks=40 --ntasks-per-node=20 --cpus-per-task=4)

printf '%s %s\n' \
    1rank_1thread "${job_1x1}" \
    4rank_1thread "${job_4x1}" \
    20rank_4thread "${job_20x4}" \
    40rank_4thread "${job_40x4}" \
    | tee "${scratch}/jobs.txt"

printf 'After all jobs finish, validate with:\n  python3 %s %s/results --markdown\n' \
    "${summarizer}" "${scratch}"
