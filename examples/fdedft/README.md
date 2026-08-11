# Frozen-density embedding DFT examples

This category contains native ABACUS FDE-DFT examples.  Each case follows the
usual example layout with committed `INPUT`, `KPT`, and `STRU` files, a
`run.sh` entry point, and compact reference output.

Available case:

- `01_f_ch3_cl_uks`: one PBE/PW91k UKS diabatic-state and coupling calculation
  for `[F-CH3-Cl]-`, using fixed alpha/beta fragment populations (the native
  two-Fermi path).

The maintained workflow also accepts PBE0-in-PBE and SCAN-in-PBE variants:
the first name is the intrafragment solver functional, while the second is the
nonadditive interfragment XC functional. See the case README for the precise
approximation and preparation commands.

These inputs demonstrate the workflow and are not converged production
benchmarks.  See the case README for resource provenance and numerical details.
The workflow writes per-call and aggregate performance JSON in addition to the
scientific output; see `tools/fde/README.md` for the field definitions.
Frozen-only semilocal functional values are cached inside each embedded SCF;
this optimization is automatic and requires no example INPUT keyword.
The workflow also provides explicit Gauss--Seidel/Jacobi updates and optional
outer linear/Anderson density mixing; the case README documents safe defaults
and MPI resource requirements.

The case now includes `euler/run_single_point.sbatch`, a scratch-isolated
20-rank x 4-thread Euler launch example. Reusable array and collection tools,
including the tested scratch layout and operational checklist, are documented
in `tools/fde/euler/README.md`.
