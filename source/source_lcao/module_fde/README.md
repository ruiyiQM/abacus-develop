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
- root artifact and state files: versioned density, determinant, band,
  fragment, and linearized-state formats;
- root embedding files: grid partitioning, semilocal functionals, `PotFde`,
  projected Hamiltonians, and the ABACUS LCAO bridge;
- root coupling/energy files: diabatic assembly, coupling, canonical ledgers,
  freeze--thaw maps, and PES utilities.

Planned migrations use `coupling/`, `gpu/`, and `workflow/` only when the
corresponding implementation is changed. Do not
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
