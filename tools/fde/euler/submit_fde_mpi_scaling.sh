#!/usr/bin/env bash

set -Eeuo pipefail

root=/cluster/home/zhourui/abacus-develop
scratch=/cluster/scratch/zhourui/abacus-fde-mpi-scaling
runner=${root}/tools/fde/euler/run_fde_mpi_scaling.sbatch

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
        "$@" "${runner}" "${layout}" "${omp_threads}"
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
    "${root}/tools/fde/euler/summarize_fde_mpi_scaling.py" "${scratch}"
