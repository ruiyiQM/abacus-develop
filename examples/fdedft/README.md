# Frozen-density embedding DFT examples

This category contains native ABACUS FDE-DFT examples. Each case has a `run.sh`
entry point and compact reference output. A single-system case keeps committed
`INPUT`, `KPT`, and `STRU` files at its root; an acceptance comparison keeps
one set below each named representation.

Available cases:

- `01_f_ch3_cl_uks`: one PBE/PW91k UKS diabatic-state and coupling calculation
  for `[F-CH3-Cl]-`, using fixed alpha/beta fragment populations (the native
  two-Fermi path).
- `02_lih_kpoint_folding`: periodic complex-k embedded SCF acceptance; compares
  primitive-cell Gamma/X eigenvalues with a `2 x 1 x 1` supercell Gamma
  spectrum for both spins.

The maintained workflow also accepts PBE0-in-PBE and SCAN-in-PBE variants:
the first name is the intrafragment solver functional, while the second is the
nonadditive interfragment XC functional. See the case README for the precise
approximation and preparation commands.

These inputs demonstrate the workflow and are not converged production
benchmarks.  See the case README for resource provenance and numerical details.
The durable cross-case regression inventory and current implementation status
are maintained in `tools/fde/regression_manifest.json` and
`source/source_lcao/module_fde/DEVELOPMENT_STATUS.md`.
The workflow writes per-call and aggregate performance JSON in addition to the
scientific output; see `tools/fde/README.md` for the field definitions.
Frozen-only semilocal functional values are cached inside each embedded SCF;
this optimization is automatic and requires no example INPUT keyword.
The workflow also provides explicit Gauss--Seidel/Jacobi updates and optional
outer linear/Anderson density mixing; the case README documents safe defaults
and MPI resource requirements.

Both cases include scratch-isolated Euler batch wrappers. Reusable array and
collection tools, including the tested scratch layout and operational
checklist, are documented in `tools/fde/euler/README.md`.
