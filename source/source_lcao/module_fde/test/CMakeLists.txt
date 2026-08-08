AddTest(
  TARGET MODULE_FDE_ao_projection
  SOURCES fde_ao_projection_test.cpp ../fde_ao_projection.cpp
)

AddTest(
  TARGET MODULE_FDE_state
  SOURCES fde_state_test.cpp ../fde_state.cpp
)

AddTest(
  TARGET MODULE_FDE_density_artifact
  SOURCES fde_density_artifact_test.cpp ../fde_density_artifact.cpp
)

AddTest(
  TARGET MODULE_FDE_semilocal_functional
  SOURCES fde_semilocal_functional_test.cpp ../fde_semilocal_functional.cpp
)

AddTest(
  TARGET MODULE_FDE_subspace_solver
  SOURCES fde_subspace_solver_test.cpp ../fde_subspace_solver.cpp ../fde_ao_projection.cpp
)

AddTest(
  TARGET MODULE_FDE_energy_ledger
  SOURCES fde_energy_ledger_test.cpp ../fde_energy_ledger.cpp
)

AddTest(
  TARGET MODULE_FDE_potential
  SOURCES fde_potential_test.cpp ../fde_potential_evaluator.cpp ../fde_semilocal_functional.cpp
)

AddTest(
  TARGET MODULE_FDE_one_way_scf
  SOURCES fde_one_way_scf_test.cpp
    ../fde_one_way_scf.cpp
    ../fde_density_artifact.cpp
    ../fde_potential_evaluator.cpp
    ../fde_semilocal_functional.cpp
    ../fde_subspace_solver.cpp
    ../fde_ao_projection.cpp
)

AddTest(
  TARGET MODULE_FDE_freeze_thaw
  SOURCES fde_freeze_thaw_test.cpp
    ../fde_freeze_thaw.cpp
    ../fde_density_artifact.cpp
    ../fde_energy_ledger.cpp
)
