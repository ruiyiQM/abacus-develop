#!/usr/bin/env bash

set -euo pipefail

readonly EXAMPLE_DIR=$(cd "$(dirname "$0")" && pwd)
readonly SETENV_FILE=${EXAMPLE_DIR}/../../SETENV

readonly ABACUS_PATH=$(awk -F "=" '$1=="ABACUS_PATH"{print $2}' "${SETENV_FILE}")
readonly ABACUS_NPROCS=$(awk -F "=" '$1=="ABACUS_NPROCS"{print $2}' "${SETENV_FILE}")
readonly ABACUS_THREADS=$(awk -F "=" '$1=="ABACUS_THREADS"{print $2}' "${SETENV_FILE}")
readonly ABACUS_EXECUTABLE=$(command -v "${ABACUS_PATH}")

cd "${EXAMPLE_DIR}"
python3 fetch_default_resources.py
python3 prepare_example.py

launcher=()
if (( ABACUS_NPROCS > 1 )); then
    launcher=(mpirun -np "${ABACUS_NPROCS}")
fi
for system in primitive supercell; do
    (
        cd "${system}"
        OMP_NUM_THREADS="${ABACUS_THREADS}" \
            "${launcher[@]}" "${ABACUS_EXECUTABLE}" > abacus.log 2>&1
        test -s result.fde_density
        test -s result.fde_kbands
    )
done

python3 ../../../tools/fde/compare_band_folding.py \
    --primitive primitive/result.fde_kbands \
    --supercell supercell/result.fde_kbands \
    --tolerance-ry 1e-6 \
    --output band_folding_result.json
echo "job succeeded!"
