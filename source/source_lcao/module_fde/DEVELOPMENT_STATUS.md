# ABACUS native FDE-DFT development status

- Status date: 2026-08-15
- Maintained branch: `fde-dft-unified`
- Status baseline: `6a5c88efb7cfe9bc7c475511601fd97f20e67ea4`

This file is the durable project handoff for native frozen-density embedding
(FDE) in ABACUS. It records what is implemented, what has been demonstrated,
which compact regression evidence is committed, and what remains. It does not
depend on `/cluster/scratch`: scratch paths and Slurm IDs below are provenance,
not the only copy of an input or accepted result.

The native C++ implementation at the status baseline is unchanged from
`411f963a0f8b2624fb83796afc83b4512601b9ad`; later commits through the status
baseline update tools, examples, documentation, and compact references.

## Executive status

The current code is a usable research implementation for two-fragment,
collinear LCAO FDE calculations and Gamma-point FDE-FODFT coupling. Its best
validated production path is PBE/PW91k UKS on molecules with explicit fixed
alpha/beta fragment populations. Distributed AO matrices and density grids,
periodic complex-k embedded SCF, RKS, three NAKE choices, persistent subsystem
sessions, adaptive freeze--thaw convergence, and CUDA kernels are implemented.

It is not yet a general-purpose subsystem-DFT implementation. In particular,
the maintained coupling artifact is molecular Gamma-only, nonadditive XC is
PBE-only, NLCC pseudopotentials are rejected, `kpar` is fixed to one, and the
multi-rank FDE FFT remains on CPU even in CUDA builds. PBE0-in-PBE and
SCAN-in-PBE are supported fragment-local approximations, not full nonadditive
hybrid or meta-GGA embedding.

## Architecture now in the tree

The module uses four primary areas plus one auxiliary orchestration layer:

| Area | Implemented responsibility |
|---|---|
| `io/` | Versioned binary/text density checkpoints and fragment, determinant, and k-point band artifacts with fingerprints and population checks. |
| `runtime/` | Sidecar parsing, explicit state/fragment occupations, solver policy, persistent-session protocol, and resident DM/orbital warm starts. |
| `embedding/` | AO projection, distributed projected Hamiltonians, PBE nonadditive XC, TF/PW91k/revAPBEk NAKE, PW differentials, LCAO/Gint integration, forces, and CUDA kernels. |
| `coupling/` | Determinant overlap, transition densities, symmetric-linearized coupling, K/L/M diabatic assembly, and nonorthogonal multistate solution. |
| `orchestration/` | Energy ledgers, freeze--thaw, arbitrary-fragment coordination primitives, PES scans, and finite-difference control. |

The detailed ownership rules are in `README.md`. User-visible theory and
runtime contracts are in `docs/developers_guide/native_fde.md`.

## Delivered capability

### Electronic structure and artifacts

- Explicit fragment atom partitions, neutral valence counts, and fixed charge
  and spin assignments for every diabatic state.
- Full supersystem local and nonlocal nuclear operators with a projected active
  AO variational space.
- RKS and UKS embedded SCF. UKS fixed populations use ABACUS' internal
  two-Fermi path; there is no user `two_fermi` INPUT keyword.
- Gamma-real and general-k complex projected Hamiltonians with `genelpa`,
  `elpa`, and `scalapack_gvx`; replicated `lapack` remains a serial reference.
- Distributed PW-slab density input/output and distributed AO matrix blocks.
- Restartable binary density checkpoints, converged fragment/determinant
  artifacts, k-resolved band artifacts, and a named canonical energy ledger.
- Explicit rejection of incompatible grid, population, pseudopotential,
  orbital, XC, KEDF, geometry, and session fingerprints.

### Functionals

- Selectable nonadditive kinetic functionals: Thomas--Fermi, LC94/PW91k, and
  revAPBEk, including variational GGA gradient/divergence terms.
- PBE nonadditive XC through Libxc.
- Fragment-local `pbe`, `pbe0`, and `scan` solvers with PBE embedding.
- Frozen semilocal functional caching inside each compatible embedded SCF.
- Deliberate rejection of non-PBE embedding XC, unsupported fragment XC names,
  and pseudopotentials carrying NLCC.

### Freeze--thaw and convergence

- Gauss--Seidel and Jacobi update maps, optional concurrent Jacobi launch, and
  outer none/linear/Anderson mixing with population projection.
- Same-branch density warm starts and persistent sessions that retain NAO/PW
  initialization, AO density matrices, and orbitals while resetting inner
  Broyden/Pulay history for every changed embedding potential.
- Residual-driven loose/medium/strict inner SCF scheduling, partial-density
  continuation, mixing recovery, and at least two strict confirmation sweeps.
- No workflow polling sleep. Slurm handles queueing; session communication is
  marker/queue based.
- Structured per-call, per-state, and total performance records.

### Coupling, multistate, and forces

- Gamma-only determinant overlaps, alpha/beta transition-density traces, raw
  `H12`, and Lowdin-orthogonalized coupling.
- Maintained `symmetric_linearized` provider with forward/reverse diagnostics,
  overlap reciprocity, transition-trace error, directional asymmetry, and a
  coupling-sensitivity estimate.
- Determinant-phase-invariant scientific comparison: one common orbital phase
  may flip overlap, H12, and signed coupling without creating a false failure.
- Nonorthogonal multistate diagonalization, root tracking, and K/L/M selection.
- Semilocal diagonal-state analytic force terms and an independent
  finite-difference validation path.

### CPU, MPI, and GPU

- OpenMP grid evaluation and distributed MPI AO/PW data paths.
- One-rank CUDA path using Gint, cuSOLVER or GPU-enabled ELPA, resident
  TF/PW91k/revAPBEk workspaces, and cuFFT for FDE gradients/divergence.
- Multi-rank CUDA keeps the distributed PW FFT on CPU but retains GPU LCAO and
  pointwise NAKE work.
- Libxc PBE evaluation is still host-side.

## Durable regression inventory

Run the complete offline gate from the repository root:

```bash
python3 tools/fde/run_regression.py --output /tmp/fde-regression.json
```

If a configured build tree contains CTest metadata, include all native gates:

```bash
python3 tools/fde/run_regression.py \
  --build-directory /absolute/path/to/build \
  --output /tmp/fde-regression-with-native.json
```

The machine-readable inventory is `tools/fde/regression_manifest.json`.

| Gate | Durable location | Scope |
|---|---|---|
| Native C++ | `test/` and `test/CMakeLists.txt` | 33 C++ test sources and 35 registered `MODULE_FDE_*` CTest gates, including four MPI executables and two Python-backed gates. |
| Workflow/tool offline tests | `tools/fde/test_*.py`, `tools/fde/workflow/test_*.py`, `tools/fde/euler/test_*.py` | Parsing, artifacts, sessions, adaptive SCF, recovery, update schemes, profiling, scientific comparison, band folding, result collection, and Slurm contracts. |
| Molecular scientific reference | `examples/fdedft/01_f_ch3_cl_uks` | PBE/PW91k UKS `[F-CH3-Cl]-` two-state convergence, coupling, provenance, and error budget. |
| Periodic embedded-SCF reference | `examples/fdedft/02_lih_kpoint_folding` | Primitive Gamma/X to doubled-supercell Gamma eigenvalue folding. |
| MPI reference | `tools/fde/euler/reference/fde_mpi_scaling_euler_2026-08-09.json` | 1, 4, 80, and 160 active-core consistency and single-step scaling. |
| CUDA reference | `tools/fde/euler/reference/fde_cuda_euler_2026-08-10.json` | TF/PW91k/revAPBEk CPU/GPU agreement, including physical nonuniform GGA densities. |
| Workflow timing baseline | `tools/fde/euler/reference/fde_workflow_performance_euler_2026-08-11.json` | Compact production/tight operational counters without raw restart data. |

At this archive update, the default offline runner passed 74 Python tests:
62 workflow/tool tests, 10 molecular-example tests, and 2 periodic-example
tests. Native CTests require a configured build and are therefore an explicit
optional runner stage rather than being silently reported as executed.

The repository intentionally does not store generated density grids, ABACUS
restart files, build directories, or full scheduler logs. The accepted compact
contract is: input/template plus pinned resource manifest, numerical tolerance,
final scientific data, convergence summary, executable/source provenance, and
the script that regenerates it.

## Accepted Euler results

### Molecular UKS two-state reference

`[F-CH3-Cl]-`, PBE/PW91k, DZP, 40 Ry, 20 MPI ranks x 4 OpenMP threads:

| Quantity | Production reference |
|---|---:|
| Reactant energy | `-92.931811287277014 Ry` |
| Product energy | `-92.524007210322964 Ry` |
| Freeze--thaw cycles | 6 / 6 |
| Final density RMS | `8.10e-19` / `1.29e-9` |
| Signed orthogonalized coupling | `0.000614741456414 Ry` |
| Coupling magnitude | `8.363983606 meV` |
| Overlap reciprocity error | `1.19e-18` |
| Maximum transition trace error | `2.13e-14` |

Production task `10378084_0` and tight-SCF task `10378084_1` used native
binary source `411f963a0` and workflow source `894ab472b`. Both electronic
calculations converged. Their archived Slurm task exit was nonzero only because
the submitted source snapshot still used the obsolete signed off-diagonal
comparator. The current phase-aware comparator passes:

| Production versus tight | Maximum delta | Budget |
|---|---:|---:|
| State energy | `2.0876e-5 Ry` | `1.0e-4 Ry` |
| Energy gap | `2.9077e-5 Ry` | `1.0e-4 Ry` |
| Overlap | `3.7872e-6` | `1.0e-5` |
| Raw H12 | `3.5899e-4 Ry` | `1.0e-3 Ry` |
| Orthogonalized coupling | `7.8083e-6 Ry` | `2.0e-5 Ry` |

### Periodic k-point acceptance

LiH job `10378057` compared two physical k points per spin in a primitive cell
with a `2 x 1 x 1` Gamma supercell. The maximum eigenvalue mismatch was
`1.2995116094e-9 Ry`, passing the `1.0e-6 Ry` gate.

This demonstrates periodic complex-k embedded SCF and k-resolved artifacts. It
does not demonstrate periodic determinant coupling.

### MPI consistency and scaling

The historical one-step Euler reference used GCC 12.2, OpenMPI 4.1.6, ELPA,
and `genelpa`:

| Layout | Active cores | SCF step | Energy delta from 1 rank | Speedup |
|---|---:|---:|---:|---:|
| 1 rank x 1 thread | 1 | 216.19 s | 0 | 1.00 |
| 4 ranks x 1 thread | 4 | 63.21 s | `8.9e-9 Ry` | 3.42 |
| 20 ranks x 4 threads | 80 | 11.64 s | `7.5e-9 Ry` | 18.57 |
| 40 ranks x 4 threads | 160 | 5.29 s | `6.9e-9 Ry` | 40.87 |

The 80-to-160-core step improved by 2.20x. This is a one-step strong-scaling
sample, not a complete freeze--thaw speedup. Its source commit predates later
session and profiling changes and must be refreshed before publication.

### CUDA consistency

Euler job `10305537` completed on one RTX 2080 Ti. Across uniform TF/PW91k/
revAPBEk and nonuniform PW91k/revAPBEk cases, the maximum CPU/GPU total-energy
difference was `5.12e-13 Ry`; the maximum NAKE-component difference was
`6.34e-12 Ry`.

Single-sample CPU/GPU wall ratios ranged from 0.70x to 1.95x. They show that
GPU acceleration is workload dependent and initialization dominated for this
small one-step problem. They are not a defensible overall GPU speedup. The job
also predates the final retained GPU workspace and must be rerun on the current
native source.

### Workflow performance snapshot

The production and tight variants each made 24 subsystem calls with no retry
and a 0.5 session-reuse fraction. Production recorded 229 SCF iterations and
4233 s of aggregate request wall; tight recorded 253 iterations and 1632 s.
They ran on different nodes, so the ratio is not a solver speedup. The durable
JSON preserves the operational baseline and the distinction between summed
profile phases and scheduler elapsed time.

## Tested capability versus scientific acceptance

The following features have deterministic unit or workflow coverage but do not
yet have a committed production-quality scientific reference:

| Capability | Present evidence | Missing acceptance |
|---|---|---|
| Closed-shell RKS | Runtime/configuration and workflow coverage; exploratory Euler points | One small converged RKS two-state or embedding reference with an explicit physical interpretation. |
| PBE0-in-PBE and SCAN-in-PBE | Capability validation and normal ABACUS fragment solver integration | Converged molecular references compared with PBE-in-PBE and checked for session/restart reproducibility. |
| All three NAKE choices | Analytic/unit tests and CUDA one-step agreement | Same-geometry converged FT sensitivity table for TF, PW91k, and revAPBEk. |
| More than two fragments | Native orchestration tests | Production workflow, canonical energy, scaling, and postprocessing benchmark. |
| Analytic diagonal forces | Unit and finite-difference infrastructure | Fully converged molecular force comparison on the current runtime path. |
| General multistate K/L/M | Algebra and artifact tests | Published multi-state benchmark with independently trusted couplings. |

## Known limitations

| Boundary | Current behavior | Required extension |
|---|---|---|
| NLCC pseudopotentials | Rejected | Store fragment-owned core densities and evaluate nonadditive XC with correct core ownership. |
| Nonadditive meta-GGA | Not implemented | Persist frozen kinetic-energy density and add its generalized-KS derivative. |
| Nonadditive hybrid exchange | Not implemented | Persist occupied density matrices and implement a nonlocal interfragment exchange operator. |
| Periodic coupling | Rejected | Define Born--von Karman determinant phase and k-conserving transition-density contracts. |
| K-point pools | `kpar = 1` | Distribute physical k points without breaking the AO/FDE communicator contract. |
| Nonorthogonal cells | Rejected | Generalize PW-slab/grid differential and artifact fingerprints. |
| Multi-rank GPU FFT | CPU distributed FFT | Integrate a decomposition-compatible GPU FFT and keep slabs/device buffers resident. |
| GPU nonadditive PBE XC | Host Libxc | Add a validated device implementation or device-capable XC backend. |
| Coupling model | Symmetric first-order linearized provider | Add and benchmark a higher-level provider; EmbASI or an external correlated active-fragment solver are possible interfaces, not implemented features. |
| Coupling derivatives | Not implemented | Differentiate determinant overlap and off-diagonal provider, then validate against finite differences. |

## Prioritized remaining work

### P0: make the current result reproducible on demand

1. Build the current branch from a clean GCC/OpenMPI/ELPA CPU tree and run all
   `MODULE_FDE_*` CTests plus `tools/fde/run_regression.py`.
2. Build the current CUDA tree and refresh the five-case CPU/GPU reference;
   verify the retained workspace and host/device transfer counters.
3. Refresh the four-layout MPI result with one immutable prepared template and
   the current binary SHA-256.
4. Add the offline runner and native CTest filter to continuous integration.
5. Copy every accepted compact result to Git/GitHub before deleting scratch;
   keep raw density/restart files only while a restart or audit needs them.

### P1: close the scientific validation gaps

1. Add one converged RKS reference and converged PBE0-in-PBE/SCAN-in-PBE
   references, each with the same restart and MPI reproducibility checks.
2. Run a common-geometry TF/PW91k/revAPBEk FT sensitivity benchmark.
3. Complete cutoff, cell-size, basis (DZP/TZDP/QZP), and diffuse-anion orbital
   convergence before treating a PES or coupling as publication quality.
4. Add an established charge-transfer coupling set and compare with trusted
   CDFT/high-level data; the current `[F-CH3-Cl]-` point is a regression, not a
   universal accuracy validation.
5. Demonstrate a diabatic crossing/Marcus-like scan with enough accepted points
   and error bars to separate model physics from SCF failure.
6. Validate diagonal forces against finite differences on a fully converged FT
   state.

### P2: complete periodic and accelerator support

1. Implement periodic determinant/coupling conventions and acceptance tests.
2. Generalize beyond `kpar = 1` and orthogonal cells.
3. Add a multi-rank distributed GPU FFT; keep density, gradient, flux, and
   potential slabs on device across FDE evaluations.
4. Remove or batch the remaining host-side PBE XC/device transfers.
5. Measure CPU and GPU before/after performance on the same node, geometry,
   convergence trajectory, binary, and resource set. Report medians and phase
   timers rather than one wall-time sample.

### P3: optional solver and ecosystem interfaces

1. Define a versioned external active-fragment solver contract before coupling
   PySCF CASSCF or EmbASI to the workflow.
2. Separate imported correlated-state data from KS density artifacts and make
   normalization, basis mapping, spin convention, and energy ownership
   explicit.
3. Add a higher-level coupling-provider interface only after a reference case
   can distinguish it scientifically from `symmetric_linearized`.

## Updating this record

For every new accepted benchmark:

1. record source commit, executable SHA-256, build stack, resources/checksums,
   Slurm job, MPI/OMP/GPU layout, and all numerical controls;
2. commit the smallest sufficient input, final TSV/JSON, convergence summary,
   tolerance, and regeneration script;
3. add the files to `tools/fde/regression_manifest.json` and extend
   `test_run_regression.py` when the schema changes;
4. rerun `tools/fde/run_regression.py` and the relevant native CTests;
5. update the accepted-results and remaining-work sections here;
6. only then consider scratch cleanup.
