# GCC/OpenMPI build on ETH Euler

These scripts build and run ABACUS with the Euler 2024-06 GNU stack:

- GCC 12.2.0
- OpenMPI 4.1.6
- OpenBLAS 0.3.24
- Netlib ScaLAPACK 2.2.0
- ELPA from the ABACUS GNU toolchain

The build rejects loaded Intel oneAPI/MKL modules and verifies that the final
binary links OpenMPI and OpenBLAS without unresolved libraries.

The scripts currently target the personal checkout
`/cluster/home/zhourui/abacus-develop` and keep all generated files below
`build-gcc-openmpi/`.

## CUDA build for FDE

`build_abacus_gcc_openmpi_cuda.sbatch` requests one Euler GPU and builds a
separate CUDA executable below `build-gcc-openmpi-cuda/`.  It reuses the
read-only GCC/OpenMPI dependency toolchain from the CPU build, enables
cuSOLVER, and compiles for Euler's Turing, Ampere, and Ada GPUs.  It does not
modify or block the CPU build directory.

```bash
sbatch euler/build_abacus_gcc_openmpi_cuda.sbatch
```

The FDE workflow should use `device: gpu` with `ks_solver: cusolver`.  Use one
MPI rank per allocated GPU.  A GPU-enabled ELPA installation may instead use
`ks_solver: elpa`; the default toolchain ELPA is CPU-only, so cuSOLVER is the
verified Euler path.

After the CUDA build succeeds, run the focused 40 Ry FDE regression with:

```bash
sbatch euler/test_fde_cuda.sbatch
```

The job prepares the committed `[F-CH3-Cl]-` UKS example on its exact FFT
grid and executes one deliberately loose electronic step for every
Thomas--Fermi, PW91k, and revAPBEk CPU/GPU pair.  It checks that each run
writes a valid FDE density checkpoint and reports the subsystem and
nonadditive energies when the one-step smoke calculation satisfies the loose
SCF threshold.  Results remain isolated below
`/cluster/scratch/$USER/abacus-fde-cuda-test-$SLURM_JOB_ID`.

## First build

Prepare a private toolchain copy once so dependency builds do not modify the
tracked source tree:

```bash
cd /cluster/home/zhourui/abacus-develop
mkdir -p build-gcc-openmpi
cp -a toolchain build-gcc-openmpi/toolchain
sbatch euler/build_abacus_gcc_openmpi.sbatch
```

The build script uses Euler's system OpenMPI, OpenBLAS, ScaLAPACK, CMake and
GCC. It builds only the missing ABACUS dependencies in the private toolchain,
then installs ABACUS below `build-gcc-openmpi/install`.

For later source-only rebuilds after the dependency toolchain is complete:

```bash
sbatch --export=ABACUS_SKIP_DEPENDENCIES=1 \
  euler/build_abacus_gcc_openmpi.sbatch
```

## Runtime environment and MPI smoke test

Load the compiled runtime in an interactive allocation or a batch script with:

```bash
source /cluster/home/zhourui/abacus-develop/euler/abacus_gcc_openmpi_env.sh
```

Verify that the executable launches through Slurm on two nodes:

```bash
sbatch euler/verify_abacus_gcc_openmpi.sbatch
```

For hybrid calculations, set `OMP_NUM_THREADS` to match
`--cpus-per-task`. OpenBLAS remains single-threaded to avoid nested
oversubscription.
