# Native FDE module boundaries

`module_fde` is being split incrementally by responsibility. Existing public
headers remain at the module root so downstream includes do not churn while
new lifecycle-sensitive code is placed in focused subdirectories.

Current boundary:

- `runtime/`: persistent worker protocol, immutable-session contract, and
  future task-local lifecycle helpers;
- `restart/`: resident AO density-matrix/orbital warm-start policy;
- `functionals/`: the fragment/embedding XC policy and future provider
  factories;
- `coupling/`: runtime provider selection/factory and coupling validation
  policy;
- `gpu/`: CUDA kernels and the per-`PotFde` resident density/FFT workspace;
- root artifact and state files: versioned density, determinant, band,
  fragment, and linearized-state formats;
- root embedding files: grid partitioning, semilocal functionals, `PotFde`,
  projected Hamiltonians, and the ABACUS LCAO bridge;
- root coupling/energy files: diabatic assembly, coupling, canonical ledgers,
  freeze--thaw maps, and PES utilities.

Planned migrations use `workflow/` only when the corresponding implementation
is changed. Do not
move unrelated files merely for directory symmetry: every move must preserve
the old include path or update all consumers and tests in one commit.

## Persistent-session invariant

One worker owns exactly one geometry, state, active fragment, AO projection,
spin population, solver layout, KEDF, and fixed INPUT/SCF/mixing signature.
Between requests only the active-density path, frozen-density paths, and output
prefix may change. `runtime/FdeSessionContract` enforces that rule before any
resident state is modified. `PotFde::reset_frozen_density` rebuilds the
frozen-functional cache while retaining its initialized PW differential/FFT
operator. The workflow creates a new worker whenever the signature changes.

`HamiltLCAO` replaces the owned potential-component list at every positive
ionic step. The FDE driver therefore checks component ownership and reattaches
a fresh `PotFde` from resident frozen density/Hartree arrays before the next
SCF. A cached raw component pointer is never dereferenced after
`Potential::pot_register` has replaced it.

This boundary is deliberate: reusing a wavefunction or mixing history across
different active AO spaces would be fast but scientifically invalid.

After the first compatible request, `restart/FdeWarmStart` maps the next
request to a positive ABACUS ionic step. LCAO therefore rebuilds request-local
Hamiltonian/grid state while retaining the resident DMK/DMR and orbital
coefficients. The new FDE density artifact remains the authoritative real-space
charge seed. ABACUS resets charge-mixing history at electronic iteration one,
so Broyden/DIIS vectors do not cross request boundaries.

## Exchange-correlation boundary

`FRAGMENT_XC` is the intrafragment ABACUS KS/GKS model and accepts `pbe`,
`pbe0`, or `scan`. `EMBEDDING_XC` is the nonadditive interfragment XC model
and currently accepts only `pbe`. Thus PBE0-in-PBE and SCAN-in-PBE retain the
ordinary ABACUS exact-exchange or kinetic-energy-density machinery inside
each active fragment, while `PotFde` continues to evaluate a density-only PBE
nonadditive XC potential between fragments. No interfragment exact exchange
or nonadditive meta-GGA tau term is implied.

Both choices are immutable within a persistent session and are recorded in
every density artifact. Mixing density artifacts from different fragment XC
models is rejected before an SCF starts.

## Coupling provider boundary

`COUPLING_PROVIDER symmetric_linearized` selects the maintained
state-specific first-order transition-energy model through
`coupling/FdeCouplingProviderFactory`. The legacy spelling `linearized` is
accepted and serialized canonically. Provider construction is now separate
from determinant overlap, K/L/M selection, and nonorthogonal diagonalization,
so a future higher-level provider can be added without changing those layers.

Every pair is evaluated in both bra/ket directions. The coupling layer checks
overlap reciprocity and the alpha/beta transition-density electron traces,
then records the directional transition-energy asymmetry and
`0.5 |S12| |E12(forward)-E21(reverse)|` as a linearization-sensitivity
estimate. That estimate is a diagnostic of the current first-order model, not
a statistical error bar.

## GPU residency boundary

Each CUDA-enabled `PotFde` owns a `FdeGpuWorkspace` through its persistent PW
differential operator. Reciprocal indices, G vectors, cuFFT plans, and allocated
buffers are uploaded or created once per session. For a full-box, single-rank
PW grid, PW91k/revAPBEk now fuse density regularization, spectral gradients,
pointwise NAKE, flux divergence, and final potential assembly on the device.
One density is uploaded and one final potential is downloaded per scalar
functional evaluation; intermediate gradients and fluxes never cross PCIe.
Thomas--Fermi uses the same resident point workspace on every GPU rank.

This does not change the multi-rank FFT boundary: distributed LCAO density
slabs continue to use ABACUS' MPI CPU PW transform, followed by GPU pointwise
NAKE work on each rank. A CUDA-aware distributed FFT requires a separate
decomposition/backend project and is not inferred from device-resident local
buffers.

Workflow-level SCF policy and phase aggregation are intentionally implemented
under `tools/fde/workflow/`. Native `FdeLcaoDriver`/`PotFde` timers remain the
fine-grained source for C++ phases, while the Python profiler owns cross-process
and cross-cycle accounting. Keeping those boundaries separate prevents a
process-wide profiler singleton from leaking state between resident sessions.
