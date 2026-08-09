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
- Use PBE norm-conserving pseudopotentials **without nonlinear core correction**.
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

## Prepare

1. Copy or symlink your no-NLCC UPF and orbital files to names used in
   `template/STRU`, or edit those filenames.
2. Run one ordinary calculation with the same 24-Angstrom cell and cutoff and
   request a charge-density cube. Only its grid header is used below.
3. Generate the five templates, exact-population uniform seed artifacts, and
   an absolute-path workflow file:

```bash
python3 prepare_example.py \
  --abacus /absolute/path/to/abacus \
  --pseudo-dir /absolute/path/to/pseudopotentials \
  --orbital-dir /absolute/path/to/orbitals \
  --grid-cube /absolute/path/to/grid_probe.cube
```

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
