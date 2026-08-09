# Native frozen-density embedding in ABACUS

This document defines the scientific and implementation contract for the
native frozen-density embedding (FDE) path in ABACUS. The first target is a
pair of charge-localized quasi-diabatic potential-energy surfaces. FDE and
fragment-orbital DFT may describe the same charge-transfer branches, but their
state energies are not expected to be numerically identical.

## RP0 scientific scope

The initial implementation is restricted to two fragments, collinear spin,
Gamma-only LCAO molecular calculations, semilocal PBE exchange-correlation,
and an explicitly selected approximate nonadditive kinetic-energy functional.
Every geometry and every quasi-diabatic state is an independent workflow. A
state is identified by one fixed integer charge and spin projection for each
fragment; it is never inferred from an integer state selector alone.

For active subsystem A with frozen subsystem B, the KSCED embedding potential
is

```text
v_emb,A = v_H[rho_B]
        + v_xc[rho_A + rho_B] - v_xc[rho_A]
        + delta Ts_nad[rho_A, rho_B] / delta rho_A.
```

The same expression with A and B exchanged defines the other freeze-thaw
step. Hartree terms use total valence density. Nonadditive kinetic terms use
valence spin densities under a documented spin-scaling convention. When a
pseudopotential has a nonlinear core correction, exchange-correlation uses
the core density owned by each fragment:

```text
Exc_nad = Exc[rho_A + rho_B + rho_core,A + rho_core,B]
        - Exc[rho_A + rho_core,A]
        - Exc[rho_B + rho_core,B].
```

`KineticFunctional::Lc94Pw91k` is the Lembarki-Chermette 1994
reparameterization of the PW91 enhancement factor, commonly called PW91k in
FDE work. The implementation uses the published/Libxc LC94 parameters and the
spin-scaling identity

```text
Ts[rho_alpha, rho_beta]
  = 1/2 Ts[2 rho_alpha] + 1/2 Ts[2 rho_beta].
```

RP2 also provides Thomas-Fermi and a Dirac-exchange nonadditive evaluator for
unit and variational-derivative tests. Dirac exchange is not the production XC
target. The production PBE nonadditive-XC adapter is introduced with the
ABACUS potential integration, where it can share ABACUS grid conventions
without changing the process-wide XC functional.

## Pseudopotential and AO-subspace boundary

An environment atom contributes both a local pseudopotential and, in general,
a nonlocal projector operator. A real-space `PotFDE` component cannot represent
the latter. The native LCAO path therefore obeys this order:

1. construct the full supersystem UnitCell and its complete Hamiltonian;
2. include local and nonlocal pseudopotentials from every nucleus;
3. select the principal H and S submatrices for the active fragment AO list;
4. solve only that active generalized eigenproblem;
5. add only environment Hartree, nonadditive XC, and nonadditive kinetic terms
   through the FDE real-space potential component.

Environment AO coefficients are excluded from the active variational space,
but environment nuclear operators are not removed. The active AO list must be
strictly increasing, unique, nonempty, and expressed in the authoritative
supersystem AO order. `ActiveAoProjection` encodes and tests this boundary.

`SubspaceSolver` is the dense reference implementation of this contract. It
solves the projected generalized eigenproblem with LAPACK, constructs the
spin-channel density as `C f C^T`, and expands that density into the
supersystem AO order with exactly zero rows and columns on environment AOs.
Electron counts are checked with `Tr(P S)`. The production
`FdeProjectedHamiltonian` applies the same projection directly to each local
2D block described by `Parallel_Orbitals`; the ordinary ABACUS LCAO solver can
therefore diagonalize the projected matrices with ELPA or ScaLAPACK without
replicating them.

## Canonical total energy

The total energy is assembled once from named terms, rather than by adding an
embedding correction to the last active-subsystem SCF energy:

```text
E_FDE = Ts[Phi_A] + Ts[Phi_B] + Ts_nad[rho_A, rho_B]
      + EH[rho_A + rho_B] + Exc[rho_A + rho_B]
      + Eext_local[rho_A + rho_B]
      + Eext_nonlocal[Phi_A, Phi_B] + Enn.
```

The ledger must retain every term in Rydberg and must be independently
recomputable from persisted subsystem artifacts. No SCF iteration may mutate
`PARAM`, `GlobalV`, or `GlobalC` to select a fragment or state.

`PotFde` is a dynamic LCAO potential component. It accepts the frozen Hartree
potential as explicit data, evaluates LC94/TF nonadditive kinetic terms, and
obtains nonadditive XC from an injected provider. It never adds a nuclear
potential. Its component energy is the cross Hartree plus nonadditive kinetic
and XC correction used during the active SCF; this is diagnostic and is not a
replacement for `CanonicalEnergy`, which sums the complete named ledger once.

## Execution and provenance

The user-facing `INPUT` deliberately contains only the runtime selector and a
sidecar path:

```text
INPUT_PARAMETERS
calculation       scf
basis_type        lcao
gamma_only        1
nspin             2
dft_functional    pbe
symmetry          0
ks_solver         genelpa
nelec             <active-fragment electron count>
nupdown           <active alpha minus beta population>
fde_task          embedded_scf
fde_config        FDE_CONFIG
```

`fde_task` accepts `none`, `embedded_scf`, and `diabatic_postprocess`. The
versioned, line-oriented `FDE_CONFIG` owns the fragment atom partition,
neutral valence-electron counts, explicit state charge/spin assignments,
active state and fragment, artifact paths, freeze-thaw controls, and K/L/M
postprocessing selections. Atom indices are zero based and refer to the STRU
supersystem order. For the two target states of `[F-CH3-Cl]-`, the model is

```text
FDE_CONFIG 1
ATOM_COUNT 6
FRAGMENT F 7 1 0
FRAGMENT CH3Cl 14 5 1 2 3 4 5
STATE reactant -1 0 2 F -1 0 CH3Cl 0 0
STATE product  -1 0 2 F  0 1 CH3Cl -1 -1
ACTIVE_STATE reactant
ACTIVE_FRAGMENT F
ACTIVE_DENSITY artifacts/reactant_F.fde_density
...
END_FDE_CONFIG
```

Thus the reactant branch contains closed-shell `F- + CH3Cl`, while the product
branch contains the spin-coupled `F(radical) + CH3Cl-` fragment assignment.
Every state/active-fragment pair is still one independent ABACUS process; an
external workflow creates those task-local sidecars and alternates them.

At runtime `FdeLcaoDriver` checks `nelec` and `nupdown` against that explicit
fragment assignment, loads the active warm-start density and all environment
density artifacts, and appends `PotFde` only after the ordinary potential
registry has been constructed. `FdeProjectedHamiltonian` then preserves the
full supersystem Hamiltonian—including every nuclear local and nonlocal
pseudopotential operator—but supplies the eigensolver with an exactly decoupled
low-energy block containing only the active fragment AOs. Inactive AOs receive
an identity metric and a high dummy eigenvalue; `nbands` is required to fit
inside the active AO dimension. The projection is performed in the local
`Parallel_Orbitals` row/column block and retains the original ScaLAPACK
descriptor. `lapack` remains valid for a replicated serial matrix;
`genelpa`, `elpa`, and `scalapack_gvx` are accepted for a distributed matrix.
The native runtime remains restricted to `kpar 1`, real Gamma-point LCAO,
`nspin 2`, and an orthogonal molecular cell.

The first executable runtime also rejects pseudopotentials with a nonzero
nonlinear core correction. The RP0 equations describe fragment-owned core
densities, but ABACUS does not yet expose that ownership to the task-local
driver. Rejecting NLCC avoids evaluating PBE nonadditivity with the complete
supersystem core density in both subsystem jobs. Use norm-conserving
pseudopotentials without NLCC for the current molecular path.

After a converged embedded SCF, the driver writes
`<OUTPUT_PREFIX>.fde_density` and `<OUTPUT_PREFIX>.fde_fragment`. The latter
contains the subsystem `etot`, the shared ion-ion term, the three embedding
energy corrections, occupied alpha/beta AO columns, the AO overlap, and the
final unprojected spin Hamiltonians. Every matrix remains in the authoritative
supersystem AO order. These versioned artifacts are sufficient to restart the
outer loop and to construct a two-state determinant and a linearized
transition-energy model without scraping human-readable ABACUS logs.

The converged density is distributed with the same z slabs as
`ModulePW::PW_Basis`. Frozen and active artifact grids are checked against that
layout and sliced to `nrxx` before entering the potential path. Gradient and
divergence operations for semilocal nonadditive functionals use the native
distributed PW FFT, while grid integrals and embedding energies are reduced
over the PW pool. At output, the density slabs are gathered back into the
canonical `xy * nz + z` artifact order. Distributed wavefunctions, overlap,
and the two spin Hamiltonians are gathered through `Cpxgemr2d`; AO rank zero
alone writes and explicitly closes both artifacts. A close failure is fatal so
an NFS quota error cannot leave a seemingly successful truncated checkpoint.

When an embedded SCF reaches `scf_nmax`, the driver instead writes only
`<OUTPUT_PREFIX>.partial.fde_density`. Its schema-2 `SCF` record includes a
false convergence flag, the completed electronic-iteration count, and the
last density residual. The checkpoint contains the mixed density that would
seed the next SCF and is independently normalized to the prescribed alpha and
beta populations. No partial fragment, Hamiltonian, orbital, or energy
artifact is written.

One ABACUS calculation solves exactly one geometry, one quasi-diabatic state,
one active subsystem, and one freeze-thaw cycle. An external restartable
workflow alternates A-in-B and B-in-A jobs. A frozen-density artifact records a
schema version; geometry, lattice, grid, pseudopotential, orbital, XC, and KEDF
fingerprints; alpha and beta valence densities; fixed electron populations;
and convergence metadata.

Warm starts are allowed only between neighboring geometries on the same state
branch. The two quasi-diabatic states must never initialize one another.

The embedded-SCF reader also accepts the compact
`FDE_UNIFORM_DENSITY_SEED 1` initialization format. It stores grid dimensions,
cell volume, exact alpha/beta populations, and one constant value per spin
instead of materializing two large grid vectors on disk. The reader expands
and validates it in memory. This format is only an intentionally uninformative
cycle-zero start: every converged update is written as a full, fingerprinted
`FDE_DENSITY_ARTIFACT`, and a seed must never cross state labels.

`tools/fde/fde_workflow.py` is the process-level owner of freeze-thaw and PES
execution. A JSON specification supplies an ABACUS command array, two
fragments, two or more explicit charge/spin states, one template calculation
directory per geometry, and initial density artifacts. The workflow creates
one isolated working directory and `FDE_CONFIG` per active-fragment update,
sets `OMP_NUM_THREADS=1` unless the caller already selected a value, and runs
the command without a shell. It checkpoints only after a complete A/B sweep.
Restart never shares a density between state labels.

`controls.allow_partial_scf` is false by default. When explicitly enabled,
the workflow may pass a schema-2 partial density to the next fragment update.
A sweep containing any partial update has no canonical energy, cannot satisfy
the outer convergence test, and cannot enter determinant or diabatic-coupling
postprocessing.

The optional inexact schedule is explicit and deterministic. The first
`inexact_freeze_thaw_cycles` sweeps use `inexact_scf_iterations` and
`inexact_scf_density_tolerance`; later sweeps use
`maximum_scf_iterations` and `scf_density_tolerance`. The workflow patches
both the task `INPUT` and `FDE_CONFIG`, so the recorded controls match the
actual embedded calculation. Final artifacts require at least two consecutive
complete strict sweeps (`strict_confirmation_cycles`) plus the outer density
and energy-change tolerances. Inexact-sweep energies are never assembled.

For the current two-fragment runtime, the canonical state energy is recovered
from the last complete sweep as

```text
E_FDE = E_A^ABACUS + E_B^ABACUS - E_nn
      + E_H^nad + T_s^nad + E_xc^nad.
```

The correction is taken from the final update of the sweep, where its active
and frozen densities are the final A/B pair. The workflow requires both jobs
to report the same `E_nn`, then applies density and energy thresholds only at
the sweep boundary. On convergence it writes one determinant and one
linearized-state artifact per state, an AO-overlap artifact, and a postprocess
`FDE_CONFIG`. More-than-two-fragment in-memory APIs remain available, but the
external production scheduler deliberately stops at two until the generalized
runtime energy recomputation is connected.

`examples/fde_f_ch3_cl` prepares a five-point illustrative SN2 scan for the
two charge-localized states. Given an ABACUS executable, no-NLCC PBE
pseudopotentials, numerical orbitals, and a cube header from the target grid,
it generates state-local compact seeds and an absolute-path workflow. The
example geometries and uniform seeds are execution scaffolding, not benchmark
reference data.

The generated postprocess sidecar adds one `LINEARIZED_STATE <state> <path>`
record per determinant. Run it with a minimal `INPUT` containing
`fde_task diabatic_postprocess` and `fde_config <path>`. This task is intercepted
before UnitCell construction, so it does not require `STRU`, orbitals, or
pseudopotentials. Rank zero reads the artifacts and writes
`<OUTPUT_PREFIX>.fde_diabatic` plus a compact TSV table; an MPI launch produces
the same single output without per-rank races.

The transition energy is a symmetric first-order FDE-diab approximation. For
each direction it linearizes the state energy around the corresponding
diagonal density,

```text
E_i[P_ij] = E_i[P_ii] + Tr[(P_ij - P_ii) H_i],
H_ij = 1/2 S_ij (E_i[P_ij] + E_j[P_ji]).
```

`H_i` is the average of the final unprojected active-fragment Hamiltonians for
state `i`. The output reports the raw nonorthogonal `H_ij`, determinant overlap,
and the symmetric two-state orthogonalized coupling
`(H_ij - S_ij(E_i+E_j)/2)/(1-S_ij^2)`. It also solves the full selected
`H B = S B E` problem. This model is deliberately identified as linearized;
a later grid transition-functional provider can replace it without changing
the determinant algebra or nonorthogonal solver.

`OneWayScf` owns the RP5 embedded-SCF loop. It reads a compatible active/frozen
artifact pair, evaluates the embedding potential, solves alpha and beta active
AO subspace problems, mixes the returned real-space density, enforces fixed
spin populations, and emits the next active artifact. Two narrow backend calls
perform real-space-potential-to-AO and AO-density-to-real-space transforms;
they are the only hooks that the production LCAO/Gint adapter must implement.

`FreezeThawWorkflow` owns the RP6 outer loop without owning an ABACUS solver.
An injected step runner performs one converged RP5 update and recomputes the
canonical ledger. The workflow alternates the two fragments, requires every
inner update to converge, and applies both a pair-density norm and a total
energy-change threshold only after a complete A/B cycle. Its deterministic
checkpoint contains both density artifacts, the full energy ledger, cycle
metadata, and residuals; restart is therefore defined only at complete-cycle
boundaries, where the two artifact cycle numbers are equal.

`TwoStatePesScan` owns the RP7 geometry axis and executes exactly two explicit
state definitions at each point. A state definition fixes each fragment's
alpha and beta populations. Warm starts are held in two separate slots, so a
state can inherit only from itself at the preceding geometry. Every returned
point must contain a converged RP6 checkpoint, matching geometry and state
identities, unchanged populations, and a localization score above the chosen
threshold. The deterministic PES table preserves both state labels even when
their energies cross and reports the minimum gap, crossing brackets, adjacent
energy changes, and changes in finite-difference slope as diagnostics.

`FiniteDifferenceForce` is the RP8 validation path for one state and one
coordinate. It runs four independent displaced geometries at `+/-h` and
`+/-h/2`; all four receive the same central-geometry checkpoint and must
reconverge the complete RP6 workflow without changing state populations or
losing localization. The reported force is the Richardson extrapolation of
the two central differences in Ry/Bohr. Their difference supplies an error
estimate and must remain below an explicit step-halving threshold. This path
validates PES derivatives but does not implement analytic FDE forces.

`AbacusGammaBackend` is the dense RP9 reference bridge. It converts each spin
channel between the dense Gamma-point AO contract and ABACUS `HContainer`
data, then delegates the real-space transforms to the production Gint calls;
that reference adapter still rejects a distributed container. The native
`FdeLcaoDriver` bypasses the dense contract and operates on the production
rank-local density grid and AO containers. `Potential::append_component`
transfers explicit ownership of a configured `PotFde` after the legacy
potential registry has run. This avoids a new global FDE selector and keeps
the frozen density and functional provider in the driver-owned object graph.

`LibxcPbeProvider` evaluates spin-polarized PBE exchange and correlation with
explicit Libxc functional identifiers and does not change ABACUS's
process-wide XC selection. Its standalone reference path uses finite
differences on a replicated orthorhombic grid. The production `PotFde` path
injects `PwGridDifferential`, so the same Libxc evaluation uses the distributed
ABACUS PW gradient and divergence. A build without Libxc reports the provider
as unavailable and fails explicitly if it is selected.

`MultiFragmentFreezeThawWorkflow` is the RP10 N-fragment outer loop. It stores
fragments in a deterministic vector, requires the caller to provide a complete
permutation of fragment indices for every sweep, and passes all other fragment
artifacts to the active update. Convergence is tested only after the complete
permutation has run. Its variable-length checkpoint and
`MultiFragmentEnergyLedger` retain one orbital-kinetic and one nonlocal-external
term per fragment while shared density terms are recorded exactly once. The
RP6 two-fragment API remains available for the already validated RP7/RP8 path.

`DiabaticDeterminantArtifact` is the RP11 coupling input. It stores occupied
real Γ-point AO coefficient columns, orbital energies, and the source fragment
of every alpha and beta orbital in the authoritative supersystem AO order. The
builder consumes the converged fragment subspace solutions, accepts only
integer occupations, and rejects overlapping fragment AO partitions. Orbitals
from the same fragment must be orthonormal in the AO metric; orbitals belonging
to different fragments are intentionally allowed to overlap because that
overlap is part of the antisymmetrized-product determinant used by FDE-diab.
The separate density artifact remains sufficient for freeze-thaw restarts.

`ElectronicCoupling` implements the RP12 determinant algebra. For each spin it
forms `M_ij = C_i^T S_AO C_j`, evaluates the determinant overlap, and constructs
the normalized transition density `P_ij = C_j M_ij^-1 C_i^T`. The state overlap
is normalized by the two determinant self norms. The injected transition-energy
provider evaluates `E[rho_ij]`; both directions are evaluated and averaged so
the returned real Γ-point Hamiltonian element is explicitly symmetric:

```text
H_ij = 1/2 S_ij (E[rho_ij] + E[rho_ji]).
```

Singular occupied-overlap matrices are rejected rather than regularized because
they make the transition density undefined.

`NonorthogonalMultistateSolver` implements RP13. It solves `H B = S B E` by
canonical orthogonalization: eigenvectors of `S` below an explicit cutoff are
discarded, the Hamiltonian is diagonalized in the retained orthonormal space,
and coefficients are transformed back to the diabatic basis. The reported
residual is evaluated in that retained space. Between geometry points, root
tracking constructs adiabatic cross overlaps from a caller-supplied diabatic
cross-overlap matrix, solves the global maximum-overlap assignment, reorders
energies, and phase-aligns coefficient columns. A minimum matched-overlap gate
turns loss of a branch into an explicit failure.

`FdeDiabaticAssembler` implements the RP14 `FDE-diab(K,L,M)` approximation.
`K` is an explicit ordered subset of quasi-diabatic states, `M` is the subset
of fragment orbital determinants retained in the coupling calculation, and
`L` is the subset whose inter-fragment occupied-orbital overlap blocks remain
active. An inter-fragment block is discarded only when neither endpoint is in
`L`; intra-fragment blocks are always retained. The assembler filters the
determinant artifacts, verifies that the `M` selection preserves alpha and beta
electron counts between selected states, builds symmetric `H` and `S` matrices,
and reports every omitted state, omitted fragment, and retained/discarded
inter-fragment block pair. Diagonal energies remain the canonical full FDE
energies; an injected transition-energy provider must carry any frozen
environment contribution omitted from the `M` determinant.

`DiagonalFdeAnalyticForce` implements the RP15 force correction for a
converged diagonal quasi-diabatic state. The driver supplies an ordinary
ABACUS base force containing the kinetic, external local/nonlocal, overlap
Pulay, ion-ion, and other non-embedding terms exactly once. RP15 adds three
separately reported density-dependent contributions: cross Hartree,
nonadditive semilocal kinetic, and nonadditive semilocal XC. For each term the
functional potential is contracted with every fragment density matrix through
the ABACUS-sign `LocalPotentialForceBackend`; `AbacusGammaBackend` implements
that contraction with `ModuleGint::cal_gint_fvl`.

For more than two fragments the implementation uses a deterministic
telescoping construction,

```text
F[rho_1 + ... + rho_N] - sum_I F[rho_I]
  = sum_(J=2)^N (F[rho_1 + ... + rho_J]
                 - F[rho_1 + ... + rho_(J-1)] - F[rho_J]).
```

The corresponding aggregate and new-fragment potentials are both contracted
at every step. This gives the derivative of the N-fragment nonadditive energy,
not the derivative of a two-fragment model in which all environment fragments
were merged. Pairwise Hartree reciprocity is checked before forces are
accepted. The formula assumes a fully converged variational freeze-thaw state
and a fixed molecular Gamma grid; RP8 remains the mandatory independent
finite-difference validation path. RP15 does not implement derivatives of
off-diagonal electronic couplings or state overlaps, hybrid-exchange response,
spinor forces, stress, or periodic k-point forces.

## Euler MPI consistency and strong scaling

`tools/fde/euler/submit_fde_mpi_scaling.sh` submits four GCC/OpenMPI/ELPA
layouts for the same native embedded-SCF input. Every case uses
`ks_solver genelpa`, `scf_nmax 1`, `scf_thr 1.0e-12`, and one OpenBLAS thread;
the test therefore compares one identical FDE SCF step rather than claiming a
fully converged production benchmark. The parser records physical input
fingerprints, energy, residual, ABACUS timing, and both active and allocated
CPU counts. The summarizer rejects mismatched `STRU`, `KPT`, or `FDE_CONFIG`
fingerprints and fails when either the energy or residual tolerance is
exceeded.

The 2026-08-09 Euler validation produced:

| Slurm job | nodes | MPI x OMP | active cores | SCF step (s) | Etot (Ry) | Delta E from 1 rank (Ry) | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10142237 | 1 | 1 x 1 | 1 | 216.19 | -62.5379739198 | 0.0 | 1.00 |
| 10142238 | 1 | 4 x 1 | 4 | 63.21 | -62.5379739109 | 8.9e-9 | 3.42 |
| 10142240 | 1 | 20 x 4 | 80 | 11.64 | -62.5379739123 | 7.5e-9 | 18.57 |
| 10142241 | 2 | 40 x 4 | 160 | 5.29 | -62.5379739129 | 6.9e-9 | 40.87 |

All four cases reported `DRHO = 1.9889`; the maximum difference from the
1-rank reference was zero at the recorded precision. The maximum energy
difference was `8.9e-9 Ry`, passing the `1.0e-8 Ry` gate. Doubling the large
layout from 80 to 160 active cores reduced the measured SCF step from 11.64 to
5.29 seconds, a 2.20x speedup for this single-step sample. The 1- and 4-rank
jobs reserved 80 CPUs per node to obtain enough memory but intentionally ran
only 1 and 4 OpenMP threads in total; the table and efficiency calculations
use active threads, while the JSON also preserves the Slurm allocation.
The exact job metadata, input hashes, source commit, and executable hash are
archived in
`tools/fde/euler/reference/fde_mpi_scaling_euler_2026-08-09.json`.

Run and summarize the test on Euler with:

```bash
bash tools/fde/euler/submit_fde_mpi_scaling.sh
python3 tools/fde/euler/summarize_fde_mpi_scaling.py \
  /cluster/scratch/zhourui/abacus-fde-mpi-scaling/results --markdown
```

## Delivery slices

- RP0: theory contract and AO-subspace pseudopotential spike.
- RP1: fragment/state definitions and frozen-density artifacts.
- RP2: explicit nonadditive kinetic and XC functional evaluation.
- RP3: active AO-subspace solver boundary and full external operator handling.
- RP4: FDE potential component and canonical energy ledger.
- RP5: one-way embedded SCF with density import/export.
- RP6: restartable freeze-thaw workflow.
- RP7: two-state geometry scan and PES diagnostics.
- RP8: finite-difference derivative of the fully converged workflow energy.
- RP9: explicit ABACUS Potential/Gint bridge and Libxc-PBE provider.
- RP10: arbitrary-fragment freeze-thaw, checkpoint, and canonical energy data.
- RP11: occupied-orbital and diabatic-determinant artifacts.
- RP12: determinant overlap, transition density, and electronic coupling.
- RP13: nonorthogonal multi-state diagonalization and root tracking.
- RP14: explicit `FDE-diab(K,L,M)` multi-state/multi-fragment assembly.
- RP15: semilocal diagonal-state analytic FDE force correction and Gint bridge.

RP10-RP15 extend the Gamma-point baseline to arbitrary fragment workflows,
determinant artifacts, electronic coupling, nonorthogonal multi-state
diagonalization, controlled multi-fragment FDE-diab approximations, and the
semilocal analytic diagonal-state force ledger. The native embedded-SCF path
supports MPI-distributed AO and PW layouts; periodic k-point sampling, hybrid
functionals, spinors, and analytic off-diagonal coupling/overlap derivatives
remain separate follow-up work.

## Acceptance gates

- A zero frozen density reduces exactly to the ordinary active-subsystem path.
- Exchanging A and B leaves a symmetric dimer result invariant.
- Integrated alpha and beta electron populations remain fixed in every cycle.
- A-first and B-first freeze-thaw orders reach the same converged energy.
- Saved artifacts reproduce the canonical energy without rerunning SCF.
- Tightening grid, SCF, and freeze-thaw thresholds changes the PES by less than
  the documented numerical error budget.
- Charge-localization diagnostics do not swap labels near a crossing.
- Each finite-difference displacement reconverges the complete freeze-thaw
  workflow before the derivative is formed.

`MODULE_FDE_workflow_serial` exercises two states, both active-fragment
updates, complete-sweep checkpointing, canonical energy assembly, automatic
postprocessing, and PES table output with a deterministic solver fixture.
`MODULE_FDE_diabatic_postprocess` covers the nonorthogonal solve. When MPI is
enabled, the PW-differential, projected-Hamiltonian, and pool-collective tests
exercise the distributed grid, local AO blocks, reductions, and canonical
density gather on two or four ranks. `MODULE_FDE_electronic_coupling_2rank`
independently evaluates the determinant overlap, transition densities, and
symmetric coupling on two ranks and requires identical finite results. The
Euler 1/4/80/160 test above is the executable production-level MPI acceptance
gate.

The quasi-diabatic-state construction follows the FDE-diab framework described
in J. Chem. Phys. 148, 214104 (2018), DOI 10.1063/1.5023290. The first
PBE/nonadditive-kinetic benchmark follows J. Phys. Chem. A 119, 2355 (2015),
DOI 10.1021/jp511275e.
