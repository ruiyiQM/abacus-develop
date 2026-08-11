# Periodic LiH FDE k-point folding acceptance

This example tests the periodic complex-k native `embedded_scf` path rather
than a molecular diabatic coupling. It runs the Li-active fragment of a
spin-polarized LiH crystal in two equivalent representations:

- a primitive cell sampled at Gamma and X;
- a `2 x 1 x 1` supercell sampled at Gamma.

For each spin, the sorted union of primitive-cell Gamma/X KS eigenvalues must
match the supercell Gamma spectrum. `compare_band_folding.py` checks all 14
eigenvalues per spin with a `1e-6 Ry` tolerance. The Euler acceptance recorded
in `reference/band_folding_result.json.ref` reached `1.30e-9 Ry` maximum error.

This is a deliberately small 40 Ry/DZP implementation check. It is not a
converged LiH band-structure calculation, and it does not exercise periodic
determinant coupling, which remains unsupported.

## Run

Set `ABACUS_PATH`, `ABACUS_NPROCS`, and `ABACUS_THREADS` in `examples/SETENV`,
then run:

```bash
bash run.sh
```

The script verifies the four pinned Li/H PP/orbital resources, generates
absolute density-artifact paths in each `FDE_CONFIG`, runs both cells, and
writes `band_folding_result.json`. To use an offline ABACUS-orbitals checkout:

```bash
python3 fetch_default_resources.py --source-root /path/to/ABACUS-orbitals
python3 prepare_example.py
```

On ETH Euler, fetch resources once on the login node and submit:

```bash
python3 fetch_default_resources.py
sbatch euler/run_acceptance.sbatch
```

The batch wrapper copies the case to a unique scratch directory. Positional
arguments may override the source root, run root, ABACUS binary, runtime setup,
and resource directory; this permits testing an uninstalled scratch build
without modifying the home checkout.

The direct embedded-SCF input uses `gamma_only 0`, `kpar 1`, UKS/two-Fermi
occupation control, PBE-in-PBE XC, and PW91k NAKE. Native FDE writes one
`result.fde_kbands` artifact per representation. K-point-parallel pools are not
enabled by this example; MPI distributes the AO/PW work within one pool.
