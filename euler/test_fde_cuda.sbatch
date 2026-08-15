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

readonly JOB_USER=${SLURM_JOB_USER:?}
readonly ABACUS_ROOT="${ABACUS_ROOT:-/cluster/home/${JOB_USER}/abacus-develop}"
readonly CUDA_BUILD_ROOT="${ABACUS_CUDA_BUILD_ROOT:-${ABACUS_ROOT}/build-gcc-openmpi-cuda}"
readonly ABACUS_BIN="${ABACUS_CUDA_BIN:-${CUDA_BUILD_ROOT}/install/bin/abacus}"
readonly RESOURCE_ROOT="${ABACUS_FDE_RESOURCE_ROOT:-${ABACUS_ROOT}/examples/fdedft/01_f_ch3_cl_uks/resources}"
readonly REFERENCE_CASE="${ABACUS_FDE_REFERENCE_CASE:-}"
readonly TOOLCHAIN_SETUP="${ABACUS_TOOLCHAIN_SETUP:-${ABACUS_ROOT}/build-gcc-openmpi/toolchain/install/setup}"
readonly CUDA_MODULE="${ABACUS_CUDA_MODULE:-cuda/13.0.2}"
readonly TEST_THREADS="${ABACUS_FDE_TEST_THREADS:-${SLURM_CPUS_PER_TASK}}"
readonly TEST_ROOT="${ABACUS_FDE_CUDA_TEST_ROOT:-/cluster/scratch/${JOB_USER}/abacus-fde-cuda-test-${SLURM_JOB_ID}}"
readonly SRUN_BIN="${ABACUS_SRUN_BIN:-/cluster/apps/slurm/bin/srun}"

if [[ -z "${REFERENCE_CASE}" ]]; then
    echo "ERROR: ABACUS_FDE_REFERENCE_CASE must name a directory containing" >&2
    echo "       a prepared direct embedded_scf FDE_CONFIG" >&2
    exit 2
fi
for required in "${ABACUS_BIN}" "${TOOLCHAIN_SETUP}" \
                "${ABACUS_ROOT}/examples/fdedft/01_f_ch3_cl_uks/prepare_example.py" \
                "${RESOURCE_ROOT}" \
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

echo "source_root=${ABACUS_ROOT}"
echo "source_commit=$(git -C "${ABACUS_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "abacus_binary=${ABACUS_BIN}"
echo "abacus_binary_sha256=$(sha256sum "${ABACUS_BIN}" | awk '{print $1}')"
echo "reference_case=${REFERENCE_CASE}"
echo "resource_root=${RESOURCE_ROOT}"
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
                "${SRUN_BIN}" --export=ALL --exclusive --ntasks=1 \
                    --cpus-per-task="${TEST_THREADS}" \
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

compare_energies()
{
    local cpu_file="$1"
    local gpu_file="$2"
    paste "${cpu_file}" "${gpu_file}" \
        | awk '
          function abs(value) { return value < 0 ? -value : value }
          $1 != $3 || abs($2 - $4) > 1.0e-8 {
              printf "energy mismatch: %s cpu=%s gpu=%s\n", $1, $2, $4 > "/dev/stderr"
              failed = 1
          }
          END { exit failed }
        '
}

for kedf in thomas_fermi pw91k revapbek; do
    compare_energies "${TEST_ROOT}/${kedf}-cpu/energies.txt" \
                     "${TEST_ROOT}/${kedf}-gpu/energies.txt"
done

# PW91k and revAPBEk also need a nonzero-gradient check.  A deliberately
# unconverged one-step CPU bootstrap turns the uniform initialization into a
# physical fragment density.  That exact checkpoint then becomes the active
# density input for both backends, exercising the CPU PW FFT versus the
# single-rank CUDA cuFFT path on identical nonuniform data.
for kedf in pw91k revapbek; do
    bootstrap_root="${TEST_ROOT}/${kedf}-bootstrap"
    mkdir -p "${bootstrap_root}"
    cp "${TEMPLATE_ROOT}/INPUT" "${TEMPLATE_ROOT}/KPT" \
       "${TEMPLATE_ROOT}/STRU" "${bootstrap_root}/"
    cp "${REFERENCE_CASE}/FDE_CONFIG" "${bootstrap_root}/FDE_CONFIG"
    cp "${ACTIVE_SEED}" "${bootstrap_root}/active.fde_seed"
    cp "${FROZEN_SEED}" "${bootstrap_root}/frozen.fde_seed"
    sed -i -E "s|^FUNCTIONALS .*|FUNCTIONALS pbe ${kedf}|" \
        "${bootstrap_root}/active.fde_seed" "${bootstrap_root}/frozen.fde_seed"
    sed -i -E \
        -e "s|^ACTIVE_DENSITY .*|ACTIVE_DENSITY ${bootstrap_root}/active.fde_seed|" \
        -e "s|^FROZEN_DENSITY CH3Cl .*|FROZEN_DENSITY CH3Cl ${bootstrap_root}/frozen.fde_seed|" \
        -e "s|^OUTPUT_PREFIX .*|OUTPUT_PREFIX result|" \
        -e "s|^KEDF .*|KEDF ${kedf}|" \
        -e "s|^MAX_SCF_ITERATIONS .*|MAX_SCF_ITERATIONS 1|" \
        -e "s|^SCF_DENSITY_TOLERANCE .*|SCF_DENSITY_TOLERANCE 1.0e-12|" \
        "${bootstrap_root}/FDE_CONFIG"
    sed -i -E \
        -e "s|^suffix[[:space:]].*|suffix                  fde_${kedf}_bootstrap|" \
        -e "s|^scf_thr[[:space:]].*|scf_thr                 1.0e-12|" \
        -e "s|^scf_nmax[[:space:]].*|scf_nmax                1|" \
        -e "s|^ks_solver[[:space:]].*|ks_solver               lapack|" \
        "${bootstrap_root}/INPUT"
    printf 'device                  cpu\n' >> "${bootstrap_root}/INPUT"
    echo "RUN ${kedf} nonuniform bootstrap"
    (
        cd "${bootstrap_root}"
        /usr/bin/time -f 'WALL_SECONDS %e' \
            "${SRUN_BIN}" --export=ALL --exclusive --ntasks=1 \
                --cpus-per-task="${TEST_THREADS}" \
                "${ABACUS_BIN}" > stdout.log 2> stderr.log
    )
    cat "${bootstrap_root}/stderr.log"
    active_checkpoint="${bootstrap_root}/result.partial.fde_density"
    if [[ ! -s "${active_checkpoint}" ]]; then
        echo "ERROR: nonuniform ${kedf} checkpoint is missing" >&2
        exit 5
    fi
    python3 - "${ABACUS_ROOT}/tools/fde" "${active_checkpoint}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from fde_workflow import read_density

density = read_density(Path(sys.argv[2]))
spreads = [max(density[spin]) - min(density[spin]) for spin in ("alpha", "beta")]
print("DENSITY_SPREAD", *spreads)
if max(spreads) <= 1.0e-8:
    raise SystemExit("bootstrap density remained uniform")
PY
    for backend in cpu gpu; do
        case_root="${TEST_ROOT}/${kedf}-nonuniform-${backend}"
        mkdir -p "${case_root}"
        cp "${TEMPLATE_ROOT}/INPUT" "${TEMPLATE_ROOT}/KPT" \
           "${TEMPLATE_ROOT}/STRU" "${case_root}/"
        cp "${REFERENCE_CASE}/FDE_CONFIG" "${case_root}/FDE_CONFIG"
        cp "${FROZEN_SEED}" "${case_root}/frozen.fde_seed"
        sed -i -E "s|^FUNCTIONALS .*|FUNCTIONALS pbe ${kedf}|" \
            "${case_root}/frozen.fde_seed"
        sed -i -E \
            -e "s|^ACTIVE_DENSITY .*|ACTIVE_DENSITY ${active_checkpoint}|" \
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
            -e "s|^suffix[[:space:]].*|suffix                  fde_${kedf}_nonuniform_${backend}|" \
            -e "s|^scf_thr[[:space:]].*|scf_thr                 5.0|" \
            -e "s|^scf_nmax[[:space:]].*|scf_nmax                1|" \
            -e "s|^ks_solver[[:space:]].*|ks_solver               ${solver}|" \
            "${case_root}/INPUT"
        printf 'device                  %s\n' "${backend}" >> "${case_root}/INPUT"

        echo "RUN ${kedf} nonuniform ${backend}"
        (
            cd "${case_root}"
            /usr/bin/time -f 'WALL_SECONDS %e' \
                "${SRUN_BIN}" --export=ALL --exclusive --ntasks=1 \
                    --cpus-per-task="${TEST_THREADS}" \
                    "${ABACUS_BIN}" > stdout.log 2> stderr.log
        )
        for artifact in result.fde_density result.fde_fragment; do
            if [[ ! -s "${case_root}/${artifact}" ]]; then
                echo "ERROR: ${artifact} is missing for nonuniform ${kedf} ${backend}" >&2
                exit 6
            fi
        done
        grep -E '^(SUBSYSTEM_TOTAL|NONADDITIVE_KINETIC|NONADDITIVE_XC)_ENERGY_RY' \
            "${case_root}/result.fde_fragment" \
            > "${case_root}/energies.txt"
        cat "${case_root}/energies.txt"
        cat "${case_root}/stderr.log"
    done
    compare_energies "${TEST_ROOT}/${kedf}-nonuniform-cpu/energies.txt" \
                     "${TEST_ROOT}/${kedf}-nonuniform-gpu/energies.txt"
done

echo "SUCCESS: ${TEST_ROOT}"
