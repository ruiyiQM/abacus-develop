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
