#!/usr/bin/env bash
#SBATCH --job-name=abacus-fde-cuda-test
#SBATCH --partition=gpu.4h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --export=NONE

set -Eeuo pipefail

readonly ABACUS_ROOT="${ABACUS_ROOT:-/cluster/home/zhourui/abacus-develop}"
readonly CUDA_BUILD_ROOT="${ABACUS_CUDA_BUILD_ROOT:-${ABACUS_ROOT}/build-gcc-openmpi-cuda}"
readonly ABACUS_BIN="${ABACUS_CUDA_BIN:-${CUDA_BUILD_ROOT}/install/bin/abacus}"
readonly RESOURCE_ROOT="${ABACUS_FDE_RESOURCE_ROOT:-/cluster/home/zhourui/abacus-develop/test_abacus/fde_f_ch3_cl/dzp/resources}"
readonly REFERENCE_CASE="${ABACUS_FDE_REFERENCE_CASE:-/cluster/home/zhourui/abacus-develop/test_abacus/fodft_parallel_validation/fodft_modes/openmpi_elpa}"
readonly TOOLCHAIN_SETUP="${ABACUS_TOOLCHAIN_SETUP:-/cluster/home/zhourui/abacus-develop/build-gcc-openmpi/toolchain/install/setup}"
readonly CUDA_MODULE="${ABACUS_CUDA_MODULE:-cuda/13.0.2}"
readonly TEST_THREADS="${ABACUS_FDE_TEST_THREADS:-${SLURM_CPUS_PER_TASK}}"
readonly TEST_ROOT="${ABACUS_FDE_CUDA_TEST_ROOT:-/cluster/scratch/${USER}/abacus-fde-cuda-test-${SLURM_JOB_ID}}"
readonly SRUN_BIN="${ABACUS_SRUN_BIN:-$(command -v srun)}"

for required in "${ABACUS_BIN}" "${TOOLCHAIN_SETUP}" \
                "${ABACUS_ROOT}/examples/fdedft/01_f_ch3_cl_uks/prepare_example.py" \
                "${REFERENCE_CASE}/FDE_CONFIG" "${SRUN_BIN}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path is missing: ${required}" >&2
        exit 2
    fi
done
if [[ ! "${TEST_THREADS}" =~ ^[1-9][0-9]*$ ]] \
   || (( TEST_THREADS > SLURM_CPUS_PER_TASK )); then
    echo "ERROR: ABACUS_FDE_TEST_THREADS must fit the allocated CPUs" >&2
    exit 2
fi

source /etc/environ.d/lmod.sh
export MODULEPATH=/cluster/software/lmods
module purge
module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load openblas/0.3.24
module load netlib-scalapack/2.2.0
module load "${CUDA_MODULE}"
export LD_RUN_PATH="${LD_RUN_PATH:-}"
source "${TOOLCHAIN_SETUP}"

export OMP_NUM_THREADS="${TEST_THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

readonly EXAMPLE_ROOT="${TEST_ROOT}/example"
mkdir -p "${TEST_ROOT}"
cp -a "${ABACUS_ROOT}/examples/fdedft/01_f_ch3_cl_uks" "${EXAMPLE_ROOT}"
python3 "${EXAMPLE_ROOT}/prepare_example.py" \
    --abacus "${ABACUS_BIN}" \
    --pseudo-dir "${RESOURCE_ROOT}/pseudopotentials" \
    --orbital-dir "${RESOURCE_ROOT}/orbitals" \
    --allow-unverified-resources \
    --kedf pw91k

readonly TEMPLATE_ROOT="${EXAMPLE_ROOT}/generated/uks_two_fermi/template"
readonly SEED_ROOT="${EXAMPLE_ROOT}/generated/uks_two_fermi/seeds"
readonly ACTIVE_SEED="${SEED_ROOT}/reactant_F.fde_seed"
readonly FROZEN_SEED="${SEED_ROOT}/reactant_CH3Cl.fde_seed"
for required in "${TEMPLATE_ROOT}/INPUT" "${TEMPLATE_ROOT}/KPT" \
                "${TEMPLATE_ROOT}/STRU" "${ACTIVE_SEED}" "${FROZEN_SEED}"; do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: prepared 40 Ry input is missing: ${required}" >&2
        exit 3
    fi
done

for kedf in thomas_fermi pw91k revapbek; do
    for backend in cpu gpu; do
        case_root="${TEST_ROOT}/${kedf}-${backend}"
        mkdir -p "${case_root}"
        cp "${TEMPLATE_ROOT}/INPUT" "${TEMPLATE_ROOT}/KPT" \
           "${TEMPLATE_ROOT}/STRU" "${case_root}/"
        cp "${REFERENCE_CASE}/FDE_CONFIG" "${case_root}/FDE_CONFIG"
        cp "${ACTIVE_SEED}" "${case_root}/active.fde_seed"
        cp "${FROZEN_SEED}" "${case_root}/frozen.fde_seed"

        sed -i -E "s|^FUNCTIONALS .*|FUNCTIONALS pbe ${kedf}|" \
            "${case_root}/active.fde_seed" "${case_root}/frozen.fde_seed"
        sed -i -E \
            -e "s|^ACTIVE_DENSITY .*|ACTIVE_DENSITY ${case_root}/active.fde_seed|" \
            -e "s|^FROZEN_DENSITY CH3Cl .*|FROZEN_DENSITY CH3Cl ${case_root}/frozen.fde_seed|" \
            -e "s|^OUTPUT_PREFIX .*|OUTPUT_PREFIX result|" \
            -e "s|^KEDF .*|KEDF ${kedf}|" \
            -e "s|^MAX_SCF_ITERATIONS .*|MAX_SCF_ITERATIONS 1|" \
            -e "s|^SCF_DENSITY_TOLERANCE .*|SCF_DENSITY_TOLERANCE 5.0|" \
            "${case_root}/FDE_CONFIG"

        if [[ "${backend}" == gpu ]]; then
            solver=cusolver
        else
            solver=lapack
        fi
        sed -i -E \
            -e "s|^suffix[[:space:]].*|suffix                  fde_${kedf}_${backend}|" \
            -e "s|^scf_thr[[:space:]].*|scf_thr                 5.0|" \
            -e "s|^scf_nmax[[:space:]].*|scf_nmax                1|" \
            -e "s|^ks_solver[[:space:]].*|ks_solver               ${solver}|" \
            "${case_root}/INPUT"
        printf 'device                  %s\n' "${backend}" >> "${case_root}/INPUT"

        echo "RUN ${kedf} ${backend}"
        (
            cd "${case_root}"
            /usr/bin/time -f 'WALL_SECONDS %e' \
                "${SRUN_BIN}" --exclusive --ntasks=1 --cpus-per-task="${TEST_THREADS}" \
                    "${ABACUS_BIN}" > stdout.log 2> stderr.log
        )
        density_path="${case_root}/result.fde_density"
        if [[ ! -s "${density_path}" ]]; then
            density_path="${case_root}/result.partial.fde_density"
        fi
        if [[ ! -s "${density_path}" ]]; then
            echo "ERROR: FDE density artifact is missing for ${kedf} ${backend}" >&2
            exit 4
        fi
        if [[ -s "${case_root}/result.fde_fragment" ]]; then
            grep -E '^(SUBSYSTEM_TOTAL|NONADDITIVE_KINETIC|NONADDITIVE_XC)_ENERGY_RY' \
                "${case_root}/result.fde_fragment" \
                > "${case_root}/energies.txt"
            cat "${case_root}/energies.txt"
        else
            head -n 14 "${density_path}" | grep -a -E '^(FUNCTIONALS|SCF|ENERGIES_RY)'
        fi
        cat "${case_root}/stderr.log"
    done
done

echo "SUCCESS: ${TEST_ROOT}"
