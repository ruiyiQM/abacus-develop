# Native FDE-DFT module

`module_fde` implements frozen-density embedding (FDE) for ABACUS LCAO
calculations and determinant-based coupling between constrained diabatic
states. The maintained end-to-end workflow is two fragments and two states:
embedded SCF → freeze–thaw → final-state energy and occupied orbitals →
overlap, Hamiltonian, effective coupling and adiabatic roots.

This README describes the code at `44ca79ae0` (2026-11-06). Engineering
regression coverage and scientific validation have different scopes; the
remaining method and convergence questions are listed below.

## Supported scope

| Path | Current implementation and boundary |
| --- | --- |
| Molecular and periodic Gamma | CPU LCAO embedded SCF, two-state freeze–thaw, final-state composition and real determinant coupling. Molecular calculations use a finite periodic box. |
| Periodic non-Gamma local transfer | One fixed finite supercell with one common boundary twist; complex occupied orbitals, AO S/H, determinant overlap and coupling are implemented. This is a single-boundary-point calculation. |
| General multi-k embedded SCF | Density and k-resolved eigenvalue export are implemented. These band records do not contain the occupied-state representation needed for local-transfer coupling on a general k mesh. |
| Spin | Collinear UKS (`nspin=2`) with prescribed fragment alpha/beta populations; closed-shell RKS (`nspin=1`) with even electron count and zero spin on every fragment. |
| Nonadditive kinetic energy | Thomas–Fermi, PW91k and revAPBEk. The compact native development cases use PBE/PW91k. |
| Exchange-correlation | Fragment XC: PBE, PBE0 or SCAN; nonadditive embedding XC: PBE with Libxc. PBE0/SCAN are fragment-local choices, without interfragment exact exchange or a nonadditive tau term. |
| CPU MPI | Distributed AO matrices and PW density grids with `kpar=1`. Focused algebra, runtime and native workflow regressions cover 1/2/4 ranks; each test has its own scope. |
| CUDA | Existing NAKE kernels and resident workspace are retained. Full-box single-rank spectral GGA work can remain on device; distributed PW FFTs use the CPU MPI path. Current CPU acceptance does not establish a new CUDA benchmark. |

Native embedded calculations require an orthogonal cell, pseudopotentials
without nonlinear core correction, `cal_force=false` and `cal_stress=false`.
Noncollinear spin, SOC, k-pool parallelism (`kpar>1`) and native force/stress
execution are outside this runtime contract. Force and finite-difference
components in the source tree do not imply an enabled force workflow.

### Meaning of periodic local transfer

The two states use the same geometry, finite supercell, explicit electron
count and `(Nalpha, Nbeta)`. One excess electron or hole is assigned to D in
one state and A in the other through fragment charge/spin constraints. The
wavefunctions describe the pseudopotential system's explicit electrons.

The small UKS hole fixture is `H2+/H2` versus `H2/H2+`, with three electrons
and `(Nalpha, Nbeta)=(2,1)`. The common twist
`controls.boundary_twist: [0.17, 0.11, 0]` selects a genuinely complex
non-Gamma path. Its components are reciprocal-cell fractions in
`[-0.5, 0.5)`; the boundary phase is `2*pi*k`. The workflow writes one KPT
point and sets `gamma_only=0`. Omitting `boundary_twist` selects the real
Gamma path; explicitly setting `[0,0,0]` exercises its complex Gamma limit.

The current workflow has exactly two fragment owners. Their densities include
the corresponding polarization, but an independently constrained third host
is not represented. Changing cell size changes periodic-image interactions;
changing twist changes the finite boundary condition. The implementation
does not multiply determinants from different twists, divide a collective
coupling by a cell count, or define a twist-averaged local coupling. General
multi-k local-state reconstruction remains unfinished. The RKS native fixture
is closed-shell pair redistribution, not the odd-electron single-hole model.

## Entry points and workflow

ABACUS `INPUT` selects a native task through `fde_task` and a sidecar through
`fde_config`:

| `fde_task` | Responsibility |
| --- | --- |
| `none` | Ordinary ABACUS calculation. |
| `embedded_scf` | One active-fragment SCF in a frozen environment. |
| `embedded_session` | A resident worker accepting compatible `RUN` requests on standard input. |
| `diabatic_postprocess` | Read final-state artifacts and assemble S/H, effective couplings and nonorthogonal roots before ordinary UnitCell setup. |

The versioned `FDE_CONFIG` sidecar carries fragment ownership, state
charge/spin assignments, XC/KEDF choices, densities, artifact paths, boundary
identity and convergence/overlap controls. An embedded job uses
`calculation=scf`, `basis_type=lcao` and `nspin=1` or `2`; its `nelec` and
`nupdown` must match the active fragment assignment. Determinant export
requires integer occupied spin orbitals and enough solved bands.

The maintained external scheduler is
[`tools/fde/fde_workflow.py`](../../../tools/fde/fde_workflow.py), rather than
the auxiliary in-memory orchestration classes. From the repository root,
validate a complete workflow specification, then execute or resume it:

```bash
python3 tools/fde/fde_workflow.py validate workflow.json
OMP_NUM_THREADS=1 python3 tools/fde/fde_workflow.py run workflow.json
```

The specification supplies command arrays, fragment/state definitions,
geometry templates and initial density artifacts. MPI launchers belong in
those command arrays. CPU solvers are `lapack` for serial AO matrices, or
`genelpa`, `elpa` and `scalapack_gvx` for distributed matrices; ELPA choices
require the corresponding build. GPU choices are `cusolver` or GPU-enabled
`elpa` with their required build dependencies.

Freeze–thaw updates active fragments using the preceding density artifacts.
Only genuinely converged SCFs and the required strict confirmation cycles
can produce final determinants. Final state energies use compatible
final-density NAKE/XC bookkeeping, rather than a raw sum of embedded SCF
energies. The workflow writes `checkpoint.json`, final determinant and
linearized-state artifacts, postprocessing results, `fde_pes.json` and
`fde_pes.tsv`; performance records describe calls and cycles separately.

Restart acceptance checks input provenance and artifact hashes. Changed or
unidentifiable inputs, incompatible artifacts and legacy checkpoints without
the required provenance are rejected; use a fresh work directory for a
changed calculation.

For the full sidecar/specification contract, see
[native FDE documentation](../../../docs/developers_guide/native_fde.md) and
the [workflow tools](../../../tools/fde/README.md). Cluster-specific build
scripts in the repository-root `euler/` directory are local operational
files, excluded from version control. The retained
[`tools/fde/euler/`](../../../tools/fde/euler/) helpers and regression records
are separate; provide site-appropriate binary and runtime-environment paths
when using them from a fresh clone.

## Source organization

| Directory | Owner and principal interfaces |
| --- | --- |
| [`io/`](io/) | Validated density, fragment, determinant and k-band formats; durable serialization, not SCF scheduling. |
| [`runtime/`](runtime/) | State/configuration, solver policy, session protocol and contract, resident warm starts. |
| [`embedding/`](embedding/) | `FdeLcaoDriver`, ABACUS/Gint integration, AO projection, distributed projected Hamiltonians, spin/grid adapters, `PotFde`, NAKE/XC and adjacent CUDA kernels. |
| [`coupling/`](coupling/) | Real/complex determinant algebra, transition densities, provider factory, linearized-state model, S/H assembly and nonorthogonal solution. |
| [`orchestration/`](orchestration/) | Auxiliary in-memory freeze–thaw, energy ledgers, multi-fragment, PES and finite-difference components. These APIs do not extend the supported external two-fragment workflow by themselves. |
| [`test/`](test/) | Algebra/unit tests, runtime failure controls, native workflow generators and CTest registration. |

ABACUS integration enters through
[`esolver_ks_lcao.cpp`](../../source_esolver/esolver_ks_lcao.cpp) and
[`read_input_item_fde.cpp`](../../source_io/module_parameter/read_input_item_fde.cpp).
The root `fde_lcao_driver.h` and `fde_diabatic_postprocess.h` are compatibility
headers; new code should include the owning subdirectory explicitly.

### Session and ownership invariants

A resident worker owns one geometry, state, active AO space, spin assignment,
solver layout, functional choice and fixed INPUT/SCF/mixing signature.
`FdeSessionContract` checks compatibility before mutating resident state;
only the request's active/frozen density paths and output prefix may vary.
The workflow creates a new worker for a changed signature.

Compatible warm requests retain AO density matrices and orbitals and use a
positive ABACUS ionic step to rebuild request-local Hamiltonian/grid state.
The density artifact remains the authoritative real-space charge seed.
ABACUS resets electronic mixing history at iteration one, so DIIS/Broyden
history does not cross requests. After potential components are replaced,
the driver reattaches `PotFde` from resident data; it does not dereference a
retired component pointer. Frozen-density updates invalidate the relevant
functional cache while keeping the initialized PW differential operator.

## Artifacts and coupling model

| Artifact | Contents and use |
| --- | --- |
| `.fde_density` | Spin density, grid/cell and resource identities, populations, functional metadata and SCF status. An unconverged job writes `.partial.fde_density`, without final coupling artifacts. |
| `.fde_fragment` | Converged fragment occupied columns in the full AO ordering, S, final unprojected spin H, energy components and density provenance. |
| `.fde_determinant` | Composed state occupied spin orbitals and identity metadata. |
| `.fde_linearized_state` | Reference energy and state-specific spin Hamiltonians for the selected provider. |
| `ao_overlap.fde_matrix` | Common AO metric for state composition and postprocessing. |
| `.fde_kbands` | Spin/k coordinates, weights and eigenvalues; no determinant data. |
| `.fde_diabatic` / `.fde_diabatic.tsv` | State overlap/Hamiltonian, pair diagnostics, effective couplings and roots. |

Fragment, determinant and linearized-state formats use real V1 or complex
V2 data. V2 binds geometry, AO/resource fingerprints and one boundary twist;
scalar versions and identities cannot be mixed. AO matrices use column-major
storage and energies use Ry. The density and k-band formats have their own
schemas; a density schema number does not select complex determinant math.

`COUPLING_PROVIDER symmetric_linearized` is the only implemented provider;
`linearized` is accepted as an alias. For each spin, full occupied overlap
forms the normalized determinant overlap and transition density using complex
bra conjugation where appropriate. The affine directional energy is

```text
E_AB^A = E_A + sum_spin Tr[H_A (P_AB - P_AA)]
E_AB   = (E_AB^A + conj(E_BA^B)) / 2
H_AB   = S_AB E_AB
```

Here `P_AA` is the normalized composed-determinant self projector, `E_A` is
the canonical final-state energy, and the workflow's `H_A` averages the final
active-fragment Hamiltonians. The implementation checks overlap reciprocity,
spin-resolved transition traces and overlap conditioning, and solves the
Hermitian nonorthogonal problem. Effective two-state coupling is reported in
the symmetric orthogonalized representation. Raw complex S/H and effective
coupling can change phase with the determinants; magnitudes and roots are the
appropriate phase-invariant comparisons.

This is an anchored affine approximation. The canonical energy uses the
sum-fragment FDE density/energy ledger; the determinant self projector includes
the inverse full occupied overlap and can differ from that density sum.
Therefore the exported reference energy and averaged H are not established
as a consistent FDE functional/gradient pair at `P_AA`. Directional asymmetry
and the reported linearization-sensitivity estimate are diagnostics, not an
error bound. One-body and bilinear Hartree energy helpers, including a native
Hartree kernel gate, exist; a complete same-density direct FDE-ET/JTD provider
and a negative-real/complex-XC prescription have not been implemented.

## Build and developer checks

Use the repository [build instructions](../../../docs/quick_start/easy_install.md)
and existing toolchains. Native embedding requires `ENABLE_LCAO=ON` and
`ENABLE_LIBXC=ON`; enable `BUILD_TESTING` for CTest. `ENABLE_JSON=ON` is needed
for JSON-dependent session checks, and `ENABLE_MPI=ON` for MPI checks. Source
changes must preserve the C++11 baseline and all applicable CMake, legacy
Make and independent pyabacus source lists.

Run the offline gate from the repository root:

```bash
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/fde/run_regression.py --output /tmp/fde-regression.json
```

It discovers the manifest's top-level, workflow, Euler-helper and example
Python suites and checks compact historical references. Those reference
checks do not regenerate molecular, MPI, CUDA or band-folding calculations.
An offline-only result has no native CTest acceptance.

With a configured and built test tree, inspect and run focused unit checks:

```bash
ctest --test-dir build -N -L fde
OMP_NUM_THREADS=1 ctest --test-dir build -R '^MODULE_FDE_' -L native_unit \
  --output-on-failure --no-tests=error
```

The labels `python`, `native_unit`, `native_runtime`, `native_workflow` and
`mpi` distinguish test scopes. A complete registered gate can be requested
with `run_regression.py --build-directory build`; it includes native workflows
and is more expensive. Read its selected/executed counts,
`coverage_complete`, unavailable tests and failures. Empty selections,
disabled/skipped tests and missing executables must not be treated as full
coverage; JSON OFF cannot establish JSON ON acceptance.

The compact native generator uses repository H pseudopotential/orbital
fixtures and temporary case directories. For example:

```bash
OMP_NUM_THREADS=1 python3 source/source_lcao/module_fde/test/test_fde_native_workflow.py \
  /absolute/path/to/abacus . --case periodic_hole_twist_uks \
  --output /tmp/fde-native-twist.json
```

Other cases include `molecule_uks`, `molecule_rks`, `periodic_gamma_uks`,
`periodic_pair_twist_rks` and `periodic_hole_boundary_consistency`. The last
compares real/complex Gamma, +/-twist and a larger cell. The corresponding
`MODULE_FDE_native_workflow_*_mpi_consistency` CTests compare fixed inputs at
1/2/4 SCF ranks with `OMP_NUM_THREADS=1`; workflow postprocessing in these
cases runs at one rank. Separate V2 CLI tests cover MPI postprocessing,
single-writer output and propagated failures. Mathematical tests cover
conjugation/Hermiticity, particle traces, allowed occupied-basis/phase
transformations, energy-zero shifts and explicit near-singular failure.

### Verification snapshot and open work

At `44ca79ae0`, Intel Make and all nine CMake cloud build jobs passed. Local
verification included 139 Python tests and two focused native CTests; a fresh
Linux JSON ON build passed 11 selected CTests and a genuine non-Gamma UKS
two-state workflow at two SCF ranks. These are selected checks, not a rerun
of every registered native case. The earlier full-suite and 1/2/4-rank
developer results belong to their recorded source snapshots. Pyabacus was
not independently built in the synchronization acceptance.

Scientific acceptance remains open: the compact charged-cell fixtures are
sensitive to cell size, twist and periodic images; general multi-k localized
states, independent host ownership and the consistent direct transition-energy
provider remain unfinished. The historical F-beta Pulay/Broyden occupied-space
and coupling discrepancy is unresolved. Current engineering checks do not
establish physical convergence, chemical accuracy or a general GPU speedup.
