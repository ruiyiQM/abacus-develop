# `[F-CH3-Cl]-` two-state FDE-diabatic scan

This example drives the native molecular Gamma-point FDE runtime through two
explicit charge-localized states:

| State | F fragment | CH3Cl fragment |
|---|---|---|
| `reactant` | charge -1, spin 0 | charge 0, spin 0 |
| `product` | charge 0, spin +1 | charge -1, spin -1 |

The atom order is always `F C H H H Cl`. Fragment `F` owns atom 0 and fragment
`CH3Cl` owns atoms 1--5. The five supplied geometries are an illustrative
collinear SN2 scan; they are starting geometries, not benchmark reference
structures.

## Requirements and scientific boundary

- Build ABACUS with LCAO and Libxc enabled.
- The reproducible default is PBE with the official SG15-v1.0
  norm-conserving pseudopotentials and matching StandardOrbitals-v2.0 DZP
  numerical atomic orbitals at 100 Ry. ABACUS calls this basis level `DZP`;
  it is the double-zeta-plus-polarization level often called DZVP by other
  codes.
- The selected H, C, F, and Cl SG15 files have no nonlinear core correction.
- Use the same pseudopotentials, numerical orbitals, cell, `ecutwfc`, and grid
  for the grid-probe calculation and every FDE job.
- The active scientific runtime is serial/replicated Gamma LCAO with
  `ks_solver lapack`. MPI is supported for artifact postprocessing and is
  covered separately; distributed active-subsystem SCF is rejected.
- `nbands 8` assumes a DZP-like F orbital basis with at least eight AOs. Adjust
  it only if it remains at least the largest active spin population and no
  larger than the F active-AO dimension.

The off-diagonal result is the documented state-specific linearized
transition-density approximation. The result table reports determinant
overlap, raw nonorthogonal `H12`, symmetric-orthogonalized coupling, and the
two generalized eigenvalues.

## Default PBE/DZP resources

The defaults are pinned by commit and SHA-256 in `default_resources.json` and
come from the official
[ABACUS-orbitals repository](https://github.com/abacusmodeling/ABACUS-orbitals).
They are downloaded into the ignored `resources/` directory, not committed to
this repository. The standard cutoff radii selected by the official set are
8 Bohr for H, C, and Cl and 7 Bohr for F.

```bash
python3 fetch_default_resources.py
```

The downloader refuses to overwrite a file with a wrong checksum unless
`--force` is explicitly supplied. An offline checkout of the official resource
repository can be used with `--source-root /path/to/ABACUS-orbitals`.

## Prepare

Generate the five geometry templates, exact-population uniform seed artifacts,
and an absolute-path workflow file:

```bash
python3 prepare_example.py --abacus /absolute/path/to/abacus
```

By default, the script verifies all eight downloaded resources and runs a
one-iteration, 22-electron spin-unpolarized ABACUS probe with 12 bands and
`out_chg 2 10`. The probe cube supplies the exact FFT grid used by the FDE seed
artifacts. Its cube
is removed after the header is read; the calculation log is retained as
`grid_probe.log`.

To reuse an existing cube made with the same `STRU`, 100 Ry cutoff, PP, and
orbitals, add `--grid-cube /absolute/path/to/grid_probe.cube`. Custom resource
directories may be supplied with `--pseudo-dir` and `--orbital-dir`; they must
use the filenames in `template/STRU`. The explicit
`--allow-unverified-resources` option skips only the pinned checksum check.

Uniform seeds are intentionally state-local but physically uninformative; the
active AO projection establishes localization during the first sweep. They are
never reused between `reactant` and `product`.

Validate and run:

```bash
python3 ../../tools/fde/fde_workflow.py validate workflow.json
OMP_NUM_THREADS=1 python3 ../../tools/fde/fde_workflow.py run workflow.json
```

Each state is checkpointed after a complete `F`/`CH3Cl` sweep. Re-running the
same command resumes incomplete states. For every geometry, inspect:

```text
work/<geometry>/<state>/checkpoint.json
work/<geometry>/postprocess/fde_diabatic.fde_diabatic.tsv
work/fde_pes.json
work/fde_pes.tsv
```

The TSV columns `h12_ry` and `orthogonalized_coupling_ry` distinguish the raw
nonorthogonal matrix element from the coupling normally plotted with the two
diabatic surfaces.
