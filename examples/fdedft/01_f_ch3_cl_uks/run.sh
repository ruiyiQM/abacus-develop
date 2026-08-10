#!/bin/bash

set -euo pipefail

EXAMPLE_DIR=$(cd "$(dirname "$0")" && pwd)
SETENV_FILE="${EXAMPLE_DIR}/../../SETENV"

ABACUS_PATH=$(awk -F "=" '$1=="ABACUS_PATH"{print $2}' "${SETENV_FILE}")
ABACUS_NPROCS=$(awk -F "=" '$1=="ABACUS_NPROCS"{print $2}' "${SETENV_FILE}")
ABACUS_THREADS=$(awk -F "=" '$1=="ABACUS_THREADS"{print $2}' "${SETENV_FILE}")
ABACUS_EXECUTABLE=$(command -v "${ABACUS_PATH}")

cd "${EXAMPLE_DIR}"
python3 fetch_default_resources.py
python3 prepare_example.py \
    --abacus "${ABACUS_EXECUTABLE}" \
    --mpi-ranks "${ABACUS_NPROCS}"
python3 ../../../tools/fde/fde_workflow.py validate workflow.json
OMP_NUM_THREADS="${ABACUS_THREADS}" \
    python3 ../../../tools/fde/fde_workflow.py run workflow.json \
    | tee fdedft.output

test -s work/fde_pes.tsv
test -s work/uks_two_fermi/postprocess/fde_diabatic.fde_diabatic.tsv
echo "job succeeded!"
