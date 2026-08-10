# Frozen-density embedding DFT examples

This category contains native ABACUS FDE-DFT examples.  Each case follows the
usual example layout with committed `INPUT`, `KPT`, and `STRU` files, a
`run.sh` entry point, and compact reference output.

Available case:

- `01_f_ch3_cl_uks`: one PBE/LC94 UKS diabatic-state and coupling calculation
  for `[F-CH3-Cl]-`, using fixed alpha/beta fragment populations (the native
  two-Fermi path).

These inputs demonstrate the workflow and are not converged production
benchmarks.  See the case README for resource provenance and numerical details.
The workflow writes per-call and aggregate performance JSON in addition to the
scientific output; see `tools/fde/README.md` for the field definitions.
