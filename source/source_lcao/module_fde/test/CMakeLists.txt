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
