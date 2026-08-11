# ABACUS FDE workflow tools

`fde_workflow.py` runs restartable frozen-density embedding freeze--thaw (FT)
calculations, constructs state-specific determinants, invokes the native ABACUS
diabatic postprocessor, and writes a potential-energy-surface table.  The
workflow uses the previous FT density artifact as the next active subsystem
initial density; an `ATOMIC` line printed during ABACUS setup does not mean that
this FDE density override was skipped.

## Commands

Validate a specification without launching ABACUS:

```bash
python3 tools/fde/fde_workflow.py validate workflow.json
```

Run or resume it:

```bash
OMP_NUM_THREADS=1 python3 tools/fde/fde_workflow.py run workflow.json
```

All external commands are supplied as JSON arrays, so MPI and Slurm launchers
are explicit and shell quoting is not reinterpreted.  The workflow contains no
polling sleeps.  Time spent by a separate queue-monitoring script is not part of
the calculation.

For the tested ETH Euler build, single-point and array runners, launcher
configuration, compact result collection, convergence playbook, and scratch
retention policy, see `tools/fde/euler/README.md`.

## Performance records

Every active-fragment directory contains `fde_performance.json` with:

- launcher wall time and return code;
- requested `scf_nmax`, `scf_thr`, and mixing controls;
- electronic-step count, first/final `drho`, and accumulated step time read from
  `OUT.*/abacus.json` when available.

Each state contains restart-safe `performance.json` and line-oriented
`performance.jsonl`.  The PES root contains `fde_performance.json`, which sums
all state calls.  These reports are rewritten from checkpoint history rather
than appended blindly, so resuming a calculation does not duplicate records.

ABACUS also reports `PotFde` timers for charge validation, active-density
preparation, functional evaluation, MPI energy reduction, effective-potential
assembly, and the complete `cal_v_eff` call.  Compare both electronic-step
counts and wall time when evaluating an SCF policy: reducing FT cycles while
making every inner solve much tighter is not necessarily a speedup.

## Nonadditive kinetic functional

Select the NAKE approximation with `controls.kedf`.  `thomas_fermi` is the
canonical name for the local Thomas--Fermi functional; the shorter `tf` spelling
is accepted as an input alias and is serialized as `thomas_fermi` by ABACUS.
It contains no density-gradient term and is therefore the lowest-cost baseline
for diagnosing whether convergence difficulty comes from a GGA kinetic
potential.  Changing the KEDF changes the model and invalidates previous FDE
density/functional-cache comparisons; use a separate work directory for each
choice.

`pw91k` is the canonical name of the default GGA NAKE.  It is the
Lembarki--Chermette 1994 reparameterization of the PW91 enhancement factor,
registered by LibXC as `GGA_K_LC94` and commonly called PW91k in FDE work.
The legacy input spelling `lc94` remains accepted, but newly generated
workflow files and deterministic `FDE_CONFIG` output use `pw91k`.

`revapbek` selects the revised APBE kinetic GGA.  Its enhancement factor uses
the LibXC `GGA_K_REVAPBE` parameters `kappa = 1.245` and `mu = 0.23889`:

```text
F(s) = 1 + kappa * mu * s^2 / (kappa + mu * s^2)
s    = |grad rho| / (2 * (3*pi^2)^(1/3) * rho^(4/3))
```

ABACUS evaluates the corresponding variational derivative, including the
divergence term, on the same distributed PW grid used by PW91k.  All three
choices use spin scaling independently for α and β densities.

| `controls.kedf` | Gradient dependence | Intended use |
|---|---:|---|
| `thomas_fermi` | no | inexpensive local baseline |
| `pw91k` | yes | established default for molecular FDE |
| `revapbek` | yes | alternative bounded PBE-form enhancement |

## GPU execution

Set both `controls.device` and a GPU-capable LCAO solver in the workflow:

```json
"device": "gpu",
"ks_solver": "cusolver"
```

`cusolver` is the recommended portable CUDA path.  `elpa` is also accepted,
but it offloads only when ABACUS is linked to an ELPA build with NVIDIA GPU
support.  The workflow rejects combinations such as `device: gpu` with
`genelpa`, so a request cannot silently run a CPU-only eigensolver.

The GPU path covers more than diagonalization:

- ABACUS' existing CUDA Gint kernels form the LCAO density and integrate the
  local effective potential;
- cuSOLVER, or GPU-enabled native ELPA, solves the projected FDE generalized
  eigenproblem;
- FDE CUDA kernels evaluate the pointwise Thomas--Fermi, PW91k, and revAPBEk
  energy, potential, and flux expressions;
- with one MPI rank, the FDE spectral gradients and divergences use cuFFT.

For multiple MPI ranks, the gradient/divergence remains on ABACUS' distributed
CPU PW FFT because the current GPU PW transform supports only a single pool
rank.  Pointwise NAKE work still runs on each rank's GPU.  Libxc PBE evaluation,
freeze--thaw orchestration, and density-artifact I/O remain on the CPU; moving
these small or library-owned sections through host/device copies would not be
a reliable acceleration.  Use one MPI rank per allocated GPU.  The per-call
`fde_performance.json` records `device` and `ks_solver` for auditability.

## Adaptive inner SCF

`adaptive_scf` selects an inner-SCF stage from the preceding FT density RMS.
The first cycle uses the first (loosest) stage.  Thresholds must decrease, and
only the final stage is marked `strict`.  `force_strict_cycle` guarantees enough
remaining cycles for the requested strict confirmations even if the outer
residual stalls.

```json
"adaptive_scf": {
  "enabled": true,
  "force_strict_cycle": 45,
  "stages": [
    {
      "name": "loose",
      "minimum_density_rms": 0.001,
      "maximum_iterations": 25,
      "density_tolerance": 0.0001,
      "strict": false
    },
    {
      "name": "medium",
      "minimum_density_rms": 0.00001,
      "maximum_iterations": 60,
      "density_tolerance": 0.00001,
      "strict": false
    },
    {
      "name": "strict",
      "minimum_density_rms": 0.0,
      "maximum_iterations": 200,
      "density_tolerance": 0.000003,
      "strict": true
    }
  ]
}
```

Set `allow_partial_scf: true` because non-strict stages are deliberately allowed
to pass a valid partial density to the next subsystem.  Final convergence still
requires `strict_confirmation_cycles` complete strict cycles, the FT density
criterion, and the energy criterion.  The legacy fixed
`inexact_freeze_thaw_cycles` schedule remains supported but cannot be combined
with `adaptive_scf`.

## Inner mixing and recovery

The following ABACUS INPUT controls can be set globally, per fragment, per
adaptive stage, or per fragment within a stage.  More specific settings win:

```text
mixing_type  mixing_beta  mixing_beta_mag  mixing_ndim  mixing_restart
mixing_dmr   mixing_gg0   mixing_gg0_mag   mixing_gg0_min
```

An optional recovery list is used only when a strict inner SCF returns a valid
partial density.  Each retry starts from that partial density in a new process,
so stale Pulay/Broyden history is discarded:

```json
"mixing_recovery": {
  "enabled": true,
  "fallbacks": [
    {"mixing_type": "pulay", "mixing_beta": 0.05,
     "mixing_beta_mag": 0.05, "mixing_ndim": 8},
    {"mixing_type": "plain", "mixing_beta": 0.03,
     "mixing_beta_mag": 0.03}
  ]
}
```

The workflow does not retry a crashed ABACUS process and does not retry a
deliberately partial loose/medium stage.  It also never carries an inner mixing
history between different embedding potentials.  Retry directories use names
such as `F-retry-01`, and all attempts are retained in performance metadata.

## Frozen functional cache

For an embedded SCF, ABACUS now prepares the frozen-only PW91k and PBE functional
values once when `PotFde` is constructed.  Every subsequent electronic step
reuses those immutable energy/potential arrays and evaluates only the total and
active densities.  UKS therefore removes one of the three frozen/active/total
functional evaluations per step.  If both active and frozen α/β densities are
bitwise equal (the native RKS path), the spin-scaled semilocal kinetic evaluator
also computes one channel and copies it to the other.

The cache is owned by one `PotFde`, so it cannot survive a change of frozen
density, grid, density floor, KEDF, or XC provider.  A new active-fragment job
constructs a new cache automatically; there is no user switch and no cache file
to manage.  ABACUS reports the one-time construction under
`PotFde prepare_frozen_cache`.  Compare its call count with
`PotFde evaluate_functionals` and the PW FFT timers when benchmarking.

The implementation intentionally does not replace gradients near the density
floor with a linear decomposition: `grad(max(rho,floor))` is not generally the
sum of separately floored gradients.  This preserves the reference functional
and potential at vacuum-grid points.

## Freeze--thaw update schemes

The validated production postprocessor still requires exactly two fragments.
Within that boundary the workflow supports:

- `auto`: currently selects Gauss--Seidel for the two-fragment case;
- `gauss_seidel`: the second fragment immediately sees the first fragment's
  new raw density;
- `jacobi`: both fragments see an immutable snapshot from the preceding FT
  cycle.

`update_order` must be a permutation of all fragment labels.  With Jacobi it
controls deterministic result ordering, not the density snapshot.  Set
`jacobi_parallelism: 1` to evaluate the Jacobi map sequentially, or `2` to
launch both subsystem commands concurrently.

Concurrent launch does not invent or divide Slurm resources.  Each
`abacus_command` must already be safe to run concurrently, for example by using
`srun --exclusive` with a per-step rank count inside an allocation large enough
for both steps.  Do not set parallelism to two if both launchers would claim the
same 80 cores.

## Outer linear and Anderson mixing

Outer mixing operates on the FT map after all subsystem calls in a cycle.  It
is distinct from ABACUS inner Pulay/Broyden charge mixing:

```json
"outer_mixing": {
  "type": "anderson",
  "beta": 0.5,
  "history": 4,
  "regularization": 1e-10,
  "apply_in_strict": false
}
```

- `none` passes the raw subsystem densities unchanged.
- `linear` applies `(1-beta) rho_old + beta rho_raw`.
- `anderson` minimizes the norm of the recent FT residual combination and then
  applies `beta` to the extrapolated map.  The current implementation uses a
  block-diagonal history: each fragment solves its own small Anderson system.

Mixed densities are projected to nonnegative values and independently
renormalized to the exact α/β populations before being written as binary FDE
artifacts.  A singular Anderson system automatically falls back to the current
linear update and records the reason in `checkpoint.json`.

`apply_in_strict` must currently be `false`.  Strict cycles therefore use raw
ABACUS densities, and the final density, fragment orbital artifact, energy, and
coupling all describe the same SCF solutions.  Anderson requires two to eight
history slots.  The workflow retains at least that many cycle directories even
when `retain_completed_cycles` is smaller, because prior input/raw density files
are the restartable history.  On large FFT grids this storage cost should be
considered before increasing the depth.

For two fragments, start with `auto` and `outer_mixing.type: none`.  Enable
linear damping when Gauss--Seidel oscillates.  Use Jacobi plus Anderson when
concurrent subsystem resources are genuinely available or when a snapshot
update is needed for reproducibility experiments.

## Tests

```bash
python3 -m unittest tools/fde/test_fde_workflow.py -v
python3 -m unittest tools/fde/test_fde_workflow_e2e.py -v
```

The example under `examples/fdedft/01_f_ch3_cl_uks` is the maintained starting
point for a complete two-state UKS calculation.

Use `compare_scientific_runs.py` with the example's
`scientific_error_budget.json` to check state energies, the diabatic gap,
overlap, raw and orthogonalized coupling, crossing brackets, and final FT
convergence. This scientific comparison is intentionally separate from the
bit-level MPI consistency test under `tools/fde/euler`.
