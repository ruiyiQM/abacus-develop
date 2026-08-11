# FDE-DFT operations on ETH Euler

This is the operational companion to `tools/fde/README.md` and
`docs/developers_guide/native_fde.md`. It consolidates the build, convergence,
Slurm, provenance, and scratch-management practices established by the Euler
FDE runs through 2026-08-11. The committed `[F-CH3-Cl]-` calculation in
`examples/fdedft/01_f_ch3_cl_uks` is the scientific starting point.

## Recommended path

1. Build the GCC/OpenMPI/ELPA executable with the scripts under `euler/`.
2. Fetch pinned PP/orbital resources once on the login node.
3. Prepare and validate one low-cost 40 Ry UKS point.
4. Run a one-point Slurm smoke test before preparing a scan.
5. Give each scan point an independent workflow JSON and Slurm array task.
6. Collect only converged PES, convergence, specifications, and provenance into
   `/cluster/home/$USER`; keep restart data in scratch until the collection has
   been checked.

Do not move an unfinished run. Workflow JSON and checkpoint files contain
absolute paths, and a move can make an otherwise restartable calculation
unusable.

## Build and verify

The verified CPU stack is GCC 12.2.0, OpenMPI 4.1.6, OpenBLAS 0.3.24,
ScaLAPACK 2.2.0, and an isolated GNU-toolchain ELPA. Intel oneAPI/MKL is
deliberately rejected. From `/cluster/home/$USER/abacus-develop`:

```bash
mkdir -p build-gcc-openmpi
cp -a toolchain build-gcc-openmpi/toolchain
sbatch euler/build_abacus_gcc_openmpi.sbatch
```

After the dependency toolchain has been built once, a source-only rebuild is:

```bash
sbatch --export=ABACUS_SKIP_DEPENDENCIES=1 \
  euler/build_abacus_gcc_openmpi.sbatch
```

Run `euler/verify_abacus_gcc_openmpi.sbatch` after a compiler, MPI, ELPA, or
linker change. Preserve the source commit, executable SHA-256, `ldd` output,
and build log with production data. A source path or filename is not a binary
identity.

The separate CUDA build and regression are:

```bash
sbatch euler/build_abacus_gcc_openmpi_cuda.sbatch
sbatch euler/test_fde_cuda.sbatch
```

The final regression in scratch job directory
`abacus-fde-cuda-test-10305537` smoke-tested all three KEDFs and verified
CPU/GPU agreement for PW91k and revAPBEk from a physical nonuniform 40 Ry
density. Uniform-density tests alone are not a sufficient GGA-NAKE gate because
their gradient is identically zero.

## Prepare one Euler workflow

Start from a validated local workflow, then replace its direct or `mpirun`
commands with explicit `srun` arrays:

```bash
python3 tools/fde/euler/configure_slurm_workflow.py workflow.local.json \
  --output workflow.euler.json \
  --abacus /cluster/home/$USER/abacus-develop/build-gcc-openmpi/install/bin/abacus \
  --ranks 20 \
  --threads 4 \
  --persistent-session \
  --work-directory /cluster/scratch/$USER/my-fde-case/work

python3 tools/fde/fde_workflow.py validate workflow.euler.json
```

The helper writes the MPI/OpenMP layout and executable path into provenance. It
does not submit a job or modify the original JSON. Pass `--device gpu`; it
selects `cusolver` unless another supported GPU solver is given. Pass
`--exclusive` when ordinary one-shot Jacobi subsystem steps will execute
concurrently. `--persistent-session` instead creates an overlapping, exact
resident step command; do not combine the resident command itself with
`--exclusive`.

The checked example can be submitted directly after its pinned resources have
been fetched:

```bash
cd /cluster/home/$USER/abacus-develop/examples/fdedft/01_f_ch3_cl_uks
python3 fetch_default_resources.py
sbatch euler/run_single_point.sbatch
```

The batch script copies the example to a new job-specific scratch directory,
probes the exact FFT grid, generates population-exact seeds, configures a
20-rank x 4-thread launcher, validates the specification, and runs the complete
two-state calculation. It never writes calculation artifacts into the source
checkout.

## One point or a Slurm array

The reusable runner accepts either one workflow JSON or a tab-separated
manifest. A single point uses:

```bash
sbatch tools/fde/euler/run_fde_workflow.sbatch \
  /cluster/scratch/$USER/my-fde-case/specs/g09.json
```

For a scan, the manifest must contain a unique `task_id` column and one of
`spec` or `spec_path`:

```text
task_id  geometry  spec
0        g00       /cluster/scratch/$USER/my-fde-case/specs/g00.json
1        g01       /cluster/scratch/$USER/my-fde-case/specs/g01.json
```

The example above is tab-separated; spaces are shown only for readability.
Submit 20 independent points with at most four running at once:

```bash
sbatch --array=0-19%4 \
  tools/fde/euler/run_fde_workflow.sbatch \
  /cluster/scratch/$USER/my-fde-case/spec_manifest.tsv
```

Command-line `sbatch` resource options may override the 20 x 4 default, but
the `srun` layout stored in every workflow JSON must match the allocation. A
reliable one-node production baseline is 20 MPI ranks x 4 OpenMP threads. It
uses all 80 physical cores while limiting MPI startup and replicated metadata.
Benchmark the real case before choosing 80 one-thread ranks or multiple nodes.

The `%4` array throttle provides concurrency; do not place a dependency chain
between the individual points. A collector may depend on the entire array:

```bash
array_job=$(sbatch --parsable --array=0-19%4 \
  tools/fde/euler/run_fde_workflow.sbatch \
  /cluster/scratch/$USER/my-fde-case/spec_manifest.tsv)
sbatch --dependency=afterok:${array_job} collect.sbatch
```

Use `afternotok` or manual inspection if partial results are valuable. The
workflow has no polling sleeps: Slurm owns queue waiting, while checkpoints
own calculation restart.

## MPI layout scaling

The scaling helper launches the same prepared direct `embedded_scf` input as
1 rank x 1 thread, 4 x 1, 20 x 4, and two-node 40 x 4 layouts. The template is
now mandatory: it must contain `INPUT`, `STRU`, `KPT`, and `FDE_CONFIG`, and all
resource/density paths referenced by those files must remain readable from the
copied scratch case. Requiring the template explicitly prevents a benchmark
from silently reusing an old `test_abacus` artifact.

```bash
bash tools/fde/euler/submit_fde_mpi_scaling.sh \
  /cluster/scratch/$USER/abacus_test/source/COMMIT \
  /cluster/scratch/$USER/abacus_test/fde-scaling-COMMIT \
  /absolute/path/to/prepared-template \
  /cluster/scratch/$USER/abacus_test/builds/COMMIT/cpu/install/bin/abacus \
  /absolute/path/to/abacus_gcc_openmpi_env.sh \
  COMMIT
```

The scratch root must not already contain `jobs.txt`. Each result records the
supplied source commit and executable SHA-256. The runner uses the maintained
`install/bin/abacus` name; legacy `abacus_std_para` paths are not inferred.
When all four jobs finish, use the exact summarizer command printed by the
submission helper.

## Collect compact final data

After every state at every point has converged:

```bash
python3 tools/fde/euler/collect_fde_results.py \
  /cluster/scratch/$USER/my-fde-case/spec_manifest.tsv \
  /cluster/home/$USER/fde-results/my-fde-case
```

The output directory must not already exist. The collector fails closed on a
missing/nonconverged state and writes:

- `fde_pes.tsv`: combined diabatic energies, overlap, and coupling;
- `convergence.tsv`: FT cycles, final residuals and performance totals;
- `points.tsv`: source paths and SHA-256 fingerprints;
- `source_manifest.tsv`, `specs/`, and `collection.json`: compact provenance.

Use `--allow-incomplete` only for a diagnostic snapshot; incomplete points are
listed but not included in `fde_pes.tsv`. Inspect the compact directory before
removing any scratch run.

## Convergence policy

### State and occupation first

- Use UKS for charge-transfer states containing odd-electron fragments. The
  `[F-CH3-Cl]-` product assignment requires UKS.
- In native `fde_task embedded_scf` with `nspin 2`, fixed alpha/beta fragment
  populations automatically use the internal two-Fermi path. Do not add an
  obsolete `two_fermi` INPUT keyword.
- RKS is valid only when every fragment in every state has spin zero and an
  even electron count. It is not a closed-shell substitute for an open-shell
  diabatic state.
- Never warm-start one diabatic state from the density of another state.

### Inner and outer loops

Use the previous density of the same fragment/state branch as the next active
subsystem start. A uniform density is only a cycle-zero seed. The workflow
then passes full population-checked binary density artifacts between cycles.

Prefer residual-driven `adaptive_scf` over a fixed number of loose cycles:

- early FT cycles: `1e-4` to `1e-5` inner density tolerance;
- intermediate cycles: approximately `1e-5`;
- final cycles: the requested strict tolerance, for example `3e-6` at 40 Ry.

Set `allow_partial_scf: true` only with this inexact strategy. A loose partial
density can advance the FT map, but that sweep has no canonical energy and can
never satisfy final convergence or enter coupling postprocessing. Require at
least two complete strict confirmation cycles.

For two fragments, Gauss--Seidel is normally the lowest-cost update: the
second fragment immediately sees the new first-fragment density. Jacobi is
useful when subsystem jobs can truly run concurrently, but usually needs
linear/Anderson outer damping. `jacobi_parallelism: 2` is unsafe when both
commands independently request all 80 cores of one 80-core allocation.

### Mixing diagnosis and recovery

Begin with conservative Broyden settings from the committed example. If the
residual oscillates or a strict solve stalls, perform one `mixing_type: plain`
diagnostic. Monotonic linear-mixing convergence is strong evidence that the
Pulay/Broyden history is the problem, not the embedding equations.

Use `mixing_recovery` to restart a strict partial density in a new ABACUS
process with Pulay and then guarded plain mixing. The new process deliberately
drops stale history. Do not retry crashes automatically and do not apply the
recovery list to planned loose partial stages.

## Where the time goes

Every FT fragment update is a fresh ABACUS process. Repeated wall time before
the electronic steps includes MPI process launch, STRU/PP/orbital parsing, NAO
and grid setup, projected-matrix setup, density scatter, and construction of a
new frozen-functional cache. It is not safe to label the observed startup
time as density-file I/O without timer evidence.

Use `fde_performance.json`, state `performance.json`, `abacus.json`, and the
`FdeLcaoDriver`/`PotFde` timers to separate:

- density read/broadcast/scatter and checkpoint gather/write;
- one-time `prepare_frozen_cache` and embedding-potential construction;
- electronic-step time and iteration count;
- process wall time outside the SCF iterations.

The workflow has no artificial sleep. Frozen PBE/NAKE values are cached within
one embedded SCF, but a cache cannot survive a changed frozen density or a new
ABACUS process. Optimize both the number of FT calls and electronic iterations;
removing density I/O alone cannot eliminate ordinary ABACUS initialization.

## CPU, MPI, and GPU boundaries

- Distributed AO matrices and PW density grids are supported with
  `genelpa`, `elpa`, or `scalapack_gvx`; use `kpar 1`.
- Periodic complex k-point embedded SCF is supported with `gamma_only 0` and a
  normal KPT mesh. K-point pools and periodic determinant coupling remain out
  of scope.
- On a CUDA build, Gint, cuSOLVER, and pointwise TF/PW91k/revAPBEk kernels run
  on GPU. A one-rank job also fuses FDE gradients, pointwise NAKE, divergence,
  and potential assembly inside a persistent cuFFT/device workspace.
- A multi-rank GPU job currently keeps the distributed FDE PW FFT on CPU; the
  pointwise NAKE and LCAO GPU work remain accelerated. Use one rank per
  allocated GPU and measure transfers/timers before assuming a speedup.
- PBE is the supported nonadditive embedding XC. Fragment-local PBE0 and SCAN
  are supported as PBE0-in-PBE and SCAN-in-PBE: exact exchange or kinetic
  energy density stays inside the active fragment solver, while no
  interfragment exact exchange or nonadditive meta-GGA `tau` term is implied.

## Scratch organization and retention

The 2026-08-11 read-only inventory of
`/cluster/scratch/zhourui/abacus_test` found more than 30 top-level source,
build, regression, PES, RKS/UKS, k-point, and XC-capability directories, with
4,429 repeated `INPUT` files. The duplication is useful during development
but should not be the production layout.

For new work, use:

```text
/cluster/scratch/$USER/abacus_test/
  source/<commit>/
  build/<commit>/cpu-or-cuda/
  runs/<case>/<date>-<commit>/
    inputs/
    specs/
    runs/
    logs/
    spec_manifest.tsv
  regressions/<name>/<job-id>/
```

Treat the source and build directories as immutable once a scan begins. One
source/build pair can serve many points with the same executable checksum.

| data | retain until | compact record |
|---|---|---|
| active `checkpoint.json`, densities, fragment artifacts | run and any restart are finished | `convergence.tsv` plus final scientific TSV |
| source/build copy | executable identity and rebuild metadata are preserved | commit, binary SHA-256, `ldd`, build log |
| successful scan directory | compact collection was verified and backed up | collector output directory |
| failed regression | root cause and useful log were recorded | job ID, failing command, relevant log excerpt |
| old PP/orbital copies | their pinned checksums and reusable cache are known | resource manifest/checksums |

Do not automate deletion from a filename pattern. First confirm that the path
is not referenced by a queued/running job or checkpoint, that the compact
collection is complete, and that no symlink/junction redirects outside the
intended run. The tools in this directory intentionally collect and audit;
they do not delete scratch data.
