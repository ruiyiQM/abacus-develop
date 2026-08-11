# `[F-CH3-Cl]-` UKS two-Fermi FDE-DFT single point

This example computes two charge-localized diabatic states and their
linearized FDE-FODFT coupling at one near-symmetric geometry:

| quantity | value |
|---|---:|
| `r(F-C)` | 2.1973684211 Angstrom |
| `r(C-Cl)` | 2.1236842105 Angstrom |
| `r(C-Cl) - r(F-C)` | -0.0736842105 Angstrom |

The atom order is `F C H H H Cl`.  Fragment `F` owns atom 0 and fragment
`CH3Cl` owns atoms 1--5.

| state | F fragment | CH3Cl fragment |
|---|---|---|
| `reactant` | charge -1, spin 0 | charge 0, spin 0 |
| `product` | charge 0, spin +1 | charge -1, spin -1 |

## Input and occupation control

`INPUT`, `KPT`, and `STRU` are the committed ABACUS input template.  The
workflow copies them into each active-fragment job and patches `nelec` and
`nupdown` from the state table.  `workflow.template.json` explicitly selects
`spin_mode: uks`.

There is intentionally no user-facing `two_fermi` keyword.  For
`fde_task embedded_scf` with `nspin 2`, ABACUS enables its internal two-Fermi
path automatically.  Thus `nupdown 0` fixes equal alpha/beta populations for
a closed-shell fragment instead of reverting to an unconstrained shared Fermi
level.  Odd-electron fragments retain the requested signed spin population.

The example uses PBE, the PW91k (LibXC LC94) nonadditive kinetic functional, SG15-v1.0 PBE
pseudopotentials, StandardOrbitals-v2.0 DZP numerical orbitals, and a 40 Ry
grid cutoff.  The cutoff was chosen for a low-cost functional test; it is not
a converged production recommendation.

## Run

Set `ABACUS_PATH`, `ABACUS_NPROCS`, and `ABACUS_THREADS` in
`examples/SETENV`, then run:

```bash
bash run.sh
```

The script downloads and verifies the eight pinned PP/orbital files, probes
the exact ABACUS FFT grid, prepares compact population-exact seed densities,
validates the workflow, and runs it.  To prepare without running:

```bash
python3 fetch_default_resources.py
python3 prepare_example.py --abacus /absolute/path/to/abacus --mpi-ranks 2
python3 ../../../tools/fde/fde_workflow.py validate workflow.json
```

On ETH Euler, fetch the pinned resources once on the login node and submit the
scratch-isolated batch wrapper:

```bash
python3 fetch_default_resources.py
sbatch euler/run_single_point.sbatch
```

It uses the GCC/OpenMPI runtime under `/cluster/home/zhourui/abacus-develop`,
copies this example to a new `/cluster/scratch/$USER/fdedft-example-$JOBID`
directory, and configures each embedded SCF as 20 MPI ranks x 4 OpenMP threads.
Pass a different source root and run root as the first and second positional
arguments when needed. The more general single-point/array runner and compact
result collector are described in `tools/fde/euler/README.md`.

To prepare independent comparisons of all three NAKE choices, reuse the same
ABACUS executable and run:

```bash
python3 prepare_example.py --abacus /absolute/path/to/abacus --kedf thomas_fermi
python3 prepare_example.py --abacus /absolute/path/to/abacus --kedf pw91k
python3 prepare_example.py --abacus /absolute/path/to/abacus --kedf revapbek
```

The default PW91k case writes `workflow.json`, `generated/`, and `work/`.
Thomas--Fermi and revAPBEk use suffixed workflow, generated, and work paths so
their checkpoints and performance reports cannot overwrite one another.

The committed reference remains a CPU/GCC/OpenMPI/ELPA calculation.  For a
CUDA build, copy `workflow.template.json` and change the two controls below;
all active-fragment jobs then receive the same GPU selection:

```json
"device": "gpu",
"ks_solver": "cusolver"
```

Use one MPI rank per allocated GPU.  With one rank the FDE PW derivatives use
cuFFT; with multiple ranks ABACUS keeps the distributed PW FFT on CPUs while
the NAKE point kernels, Gint work, and projected eigensolve use CUDA.

An offline checkout of ABACUS-orbitals can replace the download:

```bash
python3 fetch_default_resources.py \
    --source-root /path/to/ABACUS-orbitals
```

The main output files are:

```text
work/uks_two_fermi/reactant/checkpoint.json
work/uks_two_fermi/product/checkpoint.json
work/uks_two_fermi/postprocess/fde_diabatic.fde_diabatic.tsv
work/fde_pes.tsv
work/fde_performance.json
```

Each state also has `performance.json` and `performance.jsonl`, and every
active-fragment job has an `fde_performance.json`.  These files separate ABACUS
electronic-step time from process/setup overhead and record the exact SCF
threshold and mixing controls used by each call.

The example uses the residual-driven `adaptive_scf` policy.  Its loose and
medium stages may pass population-valid partial densities forward instead of
oversolving an obsolete embedding potential.  The final stage uses the stated
`3e-6` inner threshold and still requires two complete strict FT cycles.  If a
strict Broyden solve returns only a partial density, the example retries from
that density with Pulay and finally guarded plain mixing.  See
`tools/fde/README.md` for precedence, validation rules, and the complete JSON
schema used by these controls.

Frozen-only PW91k/PBE functional values are prepared once per active-fragment
SCF and then reused for its electronic steps.  The cache is in memory, is
invalidated by constructing the next fragment job, and needs no workflow or
ABACUS INPUT option.

The committed template explicitly uses `update_scheme: auto`, which selects
Gauss--Seidel for this two-fragment donor/acceptor pair, and leaves outer mixing
off to preserve the reference trajectory.  For a convergence experiment, copy
the template and replace those controls with:

```json
"update_scheme": "jacobi",
"jacobi_parallelism": 2,
"outer_mixing": {
  "type": "anderson",
  "beta": 0.5,
  "history": 4,
  "regularization": 1e-10,
  "apply_in_strict": false
}
```

The parallel setting requires an allocation and `abacus_command` that can host
two exclusive subsystem MPI steps.  On a single 80-core node use
`jacobi_parallelism: 1`; otherwise the two nominal 80-rank calls would
oversubscribe the node.  Strict cycles never mix, so the final energies and
coupling remain tied to the final raw fragment SCFs.

The reference calculation converged the reactant and product states in five
and six freeze--thaw cycles, respectively.  Its orthogonalized coupling is
`-0.0006178906742 Ry` (`-8.406830896 meV`).  Compact reference files are under
`reference/`.  Floating-point values may vary slightly with MPI layout,
libraries, and compiler, so they are comparison targets rather than bitwise
golden files.  `reference/provenance.json` records the exact source and Euler
layout used to produce them.

The off-diagonal result is the documented state-specific linearized
transition-density approximation, not an exact many-electron coupling.
