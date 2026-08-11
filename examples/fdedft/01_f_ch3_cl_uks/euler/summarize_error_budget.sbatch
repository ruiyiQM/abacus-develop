#!/usr/bin/env bash
#SBATCH --job-name=fde-budget-summary
#SBATCH --partition=normal.4h
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --export=NONE
#SBATCH --output=fde-budget-summary-%j.out

set -Eeuo pipefail

readonly BUDGET_ROOT=${1:?usage: summarize_error_budget.sbatch BUDGET_ROOT [SOURCE_ROOT]}
readonly SOURCE_ROOT=${2:-/cluster/home/zhourui/abacus-develop}
readonly SUMMARY_TOOL=${SOURCE_ROOT}/examples/fdedft/01_f_ch3_cl_uks/summarize_error_budget.py

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONDONTWRITEBYTECODE=1

python3 "${SUMMARY_TOOL}" "${BUDGET_ROOT}" --source-root "${SOURCE_ROOT}"
