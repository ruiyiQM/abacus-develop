#!/usr/bin/env bash

set -Eeuo pipefail

readonly ABACUS_ROOT="${ABACUS_ROOT:-/cluster/home/zhourui/abacus-develop}"
readonly BUILD_ROOT="${ABACUS_CUDA_BUILD_ROOT:-${ABACUS_ROOT}/build-gcc-openmpi-cuda}"
readonly TOOLCHAIN_SETUP="${ABACUS_TOOLCHAIN_SETUP:-/cluster/home/zhourui/abacus-develop/build-gcc-openmpi/toolchain/install/setup}"
readonly BUILD_JOBS="${ABACUS_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
readonly CUDA_MODULE="${ABACUS_CUDA_MODULE:-cuda/13.0.2}"
readonly CUDA_ARCHITECTURES="${ABACUS_CUDA_ARCHITECTURES:-75;80;86;89}"

if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ABACUS_BUILD_JOBS must be a positive integer" >&2
    exit 2
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && (( BUILD_JOBS > SLURM_CPUS_PER_TASK )); then
    echo "ERROR: build jobs exceed the allocated CPUs" >&2
    exit 2
fi
if [[ ! -r "${TOOLCHAIN_SETUP}" ]]; then
    echo "ERROR: dependency setup is missing: ${TOOLCHAIN_SETUP}" >&2
    exit 3
fi

source /etc/environ.d/lmod.sh
export MODULEPATH=/cluster/software/lmods
module purge
module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load openblas/0.3.24
module load netlib-scalapack/2.2.0
module load cmake/3.27.7
module load "${CUDA_MODULE}"

# The isolated GNU toolchain supplies FFTW, Libxc, ELPA, LibRI, and LibComm.
# It contains no Intel/oneAPI components and can be shared read-only by a CUDA
# source build.
export LD_RUN_PATH="${LD_RUN_PATH:-}"
source "${TOOLCHAIN_SETUP}"

for command_name in gcc g++ gfortran mpicc mpicxx mpifort cmake nvcc; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 4
    }
done

readonly CMAKE_BUILD_DIR="${BUILD_ROOT}/cmake"
readonly INSTALL_DIR="${BUILD_ROOT}/install"
mkdir -p "${CMAKE_BUILD_DIR}" "${INSTALL_DIR}" "${BUILD_ROOT}/logs"

echo "ABACUS source: ${ABACUS_ROOT}"
echo "Git commit: $(git -C "${ABACUS_ROOT}" rev-parse HEAD)"
echo "CUDA: $(nvcc --version | tail -n 1)"
echo "CUDA architectures: ${CUDA_ARCHITECTURES}"

cmake --fresh -S "${ABACUS_ROOT}" -B "${CMAKE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
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
    -DUSE_CUDA=ON \
    -DENABLE_CUSOLVERMP=OFF \
    -DBUILD_TESTING=OFF \
    -DENABLE_NATIVE_OPTIMIZATION=OFF \
    -DGIT_SUBMODULE=OFF \
    -DCOMMIT_INFO=ON \
    -DMATH_INFO=OFF \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON

cmake --build "${CMAKE_BUILD_DIR}" --parallel "${BUILD_JOBS}"
cmake --install "${CMAKE_BUILD_DIR}"

ABACUS_BIN="${INSTALL_DIR}/bin/abacus"
[[ -x "${ABACUS_BIN}" ]] || ABACUS_BIN="${INSTALL_DIR}/bin/abacus_std_para"
if [[ ! -x "${ABACUS_BIN}" ]]; then
    echo "ERROR: installed CUDA executable is missing" >&2
    exit 5
fi
"${ABACUS_BIN}" --version
ldd "${ABACUS_BIN}" > "${BUILD_ROOT}/ldd.txt"
grep -Eq 'libcudart|libcusolver' "${BUILD_ROOT}/ldd.txt" || {
    echo "ERROR: CUDA libraries are absent from the executable" >&2
    exit 6
}
echo "SUCCESS: ${ABACUS_BIN}"
