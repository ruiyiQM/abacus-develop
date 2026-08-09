#!/usr/bin/env bash

set -Eeuo pipefail

readonly ABACUS_ROOT="/cluster/home/zhourui/abacus-develop"
readonly BUILD_ROOT="${ABACUS_ROOT}/build-gcc-openmpi"
readonly CMAKE_BUILD_DIR="${BUILD_ROOT}/cmake"
readonly INSTALL_DIR="${BUILD_ROOT}/install"
readonly TOOLCHAIN_DIR="${BUILD_ROOT}/toolchain"
readonly TOOLCHAIN_SETUP="${TOOLCHAIN_DIR}/install/setup"

BUILD_JOBS="${ABACUS_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ABACUS_BUILD_JOBS must be a positive integer: ${BUILD_JOBS}" >&2
    exit 2
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && (( BUILD_JOBS > SLURM_CPUS_PER_TASK )); then
    echo "ERROR: build jobs (${BUILD_JOBS}) exceed the Slurm allocation (${SLURM_CPUS_PER_TASK})." >&2
    exit 2
fi

if ! command -v module >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source /etc/environ.d/lmod.sh
fi
export MODULEPATH="${MODULEPATH:-/cluster/software/lmods}"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load openblas/0.3.24
module load netlib-scalapack/2.2.0
module load cmake/3.27.7

: "${OPENBLAS_EULER_ROOT:?Euler OpenBLAS module did not set OPENBLAS_EULER_ROOT}"
: "${NETLIB_SCALAPACK_EULER_ROOT:?Euler ScaLAPACK module did not set NETLIB_SCALAPACK_EULER_ROOT}"

for command_name in gcc g++ gfortran mpicc mpicxx mpifort cmake; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 3
    fi
done

loaded_modules="$(module -t list 2>&1)"
if grep -Eiq '(^|/)(intel|oneapi|mkl)' <<<"${loaded_modules}"; then
    echo "ERROR: an Intel/oneAPI/MKL module is loaded unexpectedly:" >&2
    printf '%s\n' "${loaded_modules}" >&2
    exit 3
fi

if [[ ! -x "${TOOLCHAIN_DIR}/toolchain_gnu.sh" ]]; then
    echo "ERROR: isolated GNU toolchain is missing: ${TOOLCHAIN_DIR}" >&2
    echo "Run the preparation step documented in euler/build_abacus_gcc_openmpi.sbatch." >&2
    exit 4
fi

export NPROCS_OVERWRITE="${BUILD_JOBS}"
export ELPA_EXTRA_CONFIGURE_FLAGS="--without-threading-support-check-during-build --enable-runtime-threading-support-checks --enable-allow-thread-limiting"
export LD_RUN_PATH="${LD_RUN_PATH:-}"

mkdir -p "${CMAKE_BUILD_DIR}" "${INSTALL_DIR}" "${BUILD_ROOT}/logs"

echo "ABACUS source: ${ABACUS_ROOT}"
echo "Git commit: $(git -C "${ABACUS_ROOT}" rev-parse HEAD)"
echo "Build jobs: ${BUILD_JOBS}"
echo "CMake build directory: ${CMAKE_BUILD_DIR}"
echo "Install directory: ${INSTALL_DIR}"
echo "GCC: $(gcc --version | head -n 1)"
echo "OpenMPI compiler: $(mpicxx --showme:command)"
printf '%s\n' "${loaded_modules}"

# Build only the dependencies that Euler does not provide. The math libraries
# are passed as explicit paths so every component uses the exact same ABI.
if [[ "${ABACUS_SKIP_DEPENDENCIES:-0}" == "1" ]]; then
    if [[ ! -r "${TOOLCHAIN_SETUP}" ]]; then
        echo "ERROR: dependency setup is missing: ${TOOLCHAIN_SETUP}" >&2
        exit 5
    fi
else
    (
        cd "${TOOLCHAIN_DIR}"
        ./toolchain_gnu.sh -j "${BUILD_JOBS}" \
            --target-cpu=generic \
            --skip-system-checks \
            --with-gcc=system \
            --with-openmpi=system \
            --with-cmake=system \
            --with-openblas="${OPENBLAS_EULER_ROOT}" \
            --with-fftw=install \
            --with-scalapack="${NETLIB_SCALAPACK_EULER_ROOT}" \
            --package-version elpa:alt
    )
fi

if [[ ! -r "${TOOLCHAIN_SETUP}" ]]; then
    echo "ERROR: toolchain did not create ${TOOLCHAIN_SETUP}" >&2
    exit 6
fi
# shellcheck source=/dev/null
source "${TOOLCHAIN_SETUP}"

: "${LIBRI_ROOT:?LIBRI_ROOT is missing after loading the toolchain}"
: "${LIBCOMM_ROOT:?LIBCOMM_ROOT is missing after loading the toolchain}"
: "${FFTW_ROOT:?FFTW_ROOT is missing after loading the toolchain}"

cmake --fresh -S "${ABACUS_ROOT}" -B "${CMAKE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DMPI_C_COMPILER=mpicc \
    -DMPI_CXX_COMPILER=mpicxx \
    -DMPI_Fortran_COMPILER=mpifort \
    -DLAPACK_DIR="${OPENBLAS_EULER_ROOT}/lib" \
    -DSCALAPACK_DIR="${NETLIB_SCALAPACK_EULER_ROOT}/lib" \
    -DFFTW3_DIR="${FFTW_ROOT}" \
    -DENABLE_MPI=ON \
    -DENABLE_OPENMP=ON \
    -DENABLE_FLOAT_FFTW=ON \
    -DENABLE_LCAO=ON \
    -DENABLE_ELPA=ON \
    -DENABLE_LIBXC=ON \
    -DENABLE_DFTD4=ON \
    -DENABLE_RAPIDJSON=ON \
    -DENABLE_LIBRI=ON \
    -DLIBRI_DIR="${LIBRI_ROOT}" \
    -DLIBCOMM_DIR="${LIBCOMM_ROOT}" \
    -DUSE_CUDA=OFF \
    -DBUILD_TESTING=OFF \
    -DENABLE_NATIVE_OPTIMIZATION=OFF \
    -DGIT_SUBMODULE=OFF \
    -DCOMMIT_INFO=ON \
    -DMATH_INFO=OFF \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON

cmake --build "${CMAKE_BUILD_DIR}" --parallel "${BUILD_JOBS}"
cmake --install "${CMAKE_BUILD_DIR}"

readonly ABACUS_BIN="${INSTALL_DIR}/bin/abacus"
if [[ ! -x "${ABACUS_BIN}" ]]; then
    echo "ERROR: ABACUS executable was not installed: ${ABACUS_BIN}" >&2
    exit 7
fi

"${ABACUS_BIN}" --version
ldd "${ABACUS_BIN}" | tee "${BUILD_ROOT}/ldd.txt"
if grep -q "not found" "${BUILD_ROOT}/ldd.txt"; then
    echo "ERROR: unresolved shared-library dependency" >&2
    exit 8
fi
if grep -Eiq 'mkl|oneapi|intelmpi|impi' "${BUILD_ROOT}/ldd.txt"; then
    echo "ERROR: the GCC build unexpectedly links an Intel/oneAPI library" >&2
    exit 9
fi
if ! grep -q 'libmpi' "${BUILD_ROOT}/ldd.txt" || ! grep -q 'libopenblas' "${BUILD_ROOT}/ldd.txt"; then
    echo "ERROR: expected OpenMPI/OpenBLAS libraries are absent from ldd output" >&2
    exit 10
fi

echo "SUCCESS: ${ABACUS_BIN}"
