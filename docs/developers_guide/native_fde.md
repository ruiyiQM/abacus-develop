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

`SubspaceSolver` implements the first executable form of this contract for a
replicated real Gamma-point matrix. It solves the projected generalized
eigenproblem with LAPACK, constructs the spin-channel density as `C f C^T`,
and expands that density into the supersystem AO order with exactly zero rows
and columns on environment AOs. Electron counts are checked with `Tr(P S)`.
Distributed ScaLAPACK/ELPA projection is deliberately deferred until the
serial scientific path is validated.

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

One ABACUS calculation solves exactly one geometry, one quasi-diabatic state,
one active subsystem, and one freeze-thaw cycle. An external restartable
workflow alternates A-in-B and B-in-A jobs. A frozen-density artifact records a
schema version; geometry, lattice, grid, pseudopotential, orbital, XC, and KEDF
fingerprints; alpha and beta valence densities; fixed electron populations;
and convergence metadata.

Warm starts are allowed only between neighboring geometries on the same state
branch. The two quasi-diabatic states must never initialize one another.

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

`AbacusGammaBackend` is the RP9 native runtime bridge. It converts each spin
channel between the dense Γ-point AO contract and ABACUS `HContainer` data,
then delegates the real-space transforms to the production Gint calls. The
current dense AO contract is serial by construction, so the adapter rejects a
distributed 2D-block AO container instead of silently assembling an incomplete
matrix. `Potential::append_component` transfers explicit ownership of a
configured `PotFde` after the legacy potential registry has run. This avoids a
new global FDE selector and keeps the frozen density and functional provider in
the driver-owned object graph.

`LibxcPbeProvider` evaluates spin-polarized PBE exchange and correlation with
explicit Libxc functional identifiers. It forms the GGA functional derivative
on the same orthorhombic replicated grid used by the RP0-RP8 prototype and does
not change ABACUS's process-wide XC selection. A build without Libxc reports the
provider as unavailable and fails explicitly if it is selected. General-cell,
distributed-grid PBE must use the later native PW-gradient driver rather than
this replicated-grid adapter.

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

RP10-RP15 extend this serial Γ-point baseline to arbitrary fragment workflows,
determinant artifacts, electronic coupling, nonorthogonal multi-state
diagonalization, controlled multi-fragment FDE-diab approximations, and the
semilocal analytic diagonal-state force ledger. Periodic k-point sampling,
hybrid functionals, spinors, and analytic off-diagonal coupling/overlap
derivatives remain separate follow-up work.

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

The quasi-diabatic-state construction follows the FDE-diab framework described
in J. Chem. Phys. 148, 214104 (2018), DOI 10.1063/1.5023290. The first
PBE/nonadditive-kinetic benchmark follows J. Phys. Chem. A 119, 2355 (2015),
DOI 10.1021/jp511275e.
