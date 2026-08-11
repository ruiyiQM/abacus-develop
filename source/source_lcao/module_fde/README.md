# Native FDE module boundaries

`module_fde` is being split incrementally by responsibility. Existing public
headers remain at the module root so downstream includes do not churn while
new lifecycle-sensitive code is placed in focused subdirectories.

Current boundary:

- `runtime/`: persistent worker protocol, immutable-session contract, and
  future task-local lifecycle helpers;
- root artifact and state files: versioned density, determinant, band,
  fragment, and linearized-state formats;
- root embedding files: grid partitioning, semilocal functionals, `PotFde`,
  projected Hamiltonians, and the ABACUS LCAO bridge;
- root coupling/energy files: diabatic assembly, coupling, canonical ledgers,
  freeze--thaw maps, and PES utilities.

Planned migrations use `restart/`, `functionals/`, `coupling/`, `gpu/`, and
`workflow/` only when the corresponding implementation is changed. Do not
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
