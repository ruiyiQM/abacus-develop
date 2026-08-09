AddTest(
  TARGET MODULE_FDE_ao_projection
  SOURCES fde_ao_projection_test.cpp ../fde_ao_projection.cpp
)

AddTest(
  TARGET MODULE_FDE_abacus_gamma_backend
  LIBS parameter ${math_libs} psi base device
  SOURCES fde_abacus_gamma_backend_test.cpp
    ../fde_abacus_gamma_backend.cpp
    ../../../source_hamilt/module_hcontainer/base_matrix.cpp
    ../../../source_hamilt/module_hcontainer/hcontainer.cpp
    ../../../source_hamilt/module_hcontainer/atom_pair.cpp
    ../../../source_hamilt/module_hcontainer/test/tmp_mocks.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp
)

AddTest(
  TARGET MODULE_FDE_analytic_force
  SOURCES fde_analytic_force_test.cpp
    ../fde_analytic_force.cpp
    ../fde_potential_evaluator.cpp
    ../fde_semilocal_functional.cpp
)

AddTest(
  TARGET MODULE_FDE_state
  SOURCES fde_state_test.cpp ../fde_state.cpp
)

AddTest(
  TARGET MODULE_FDE_runtime_config
  SOURCES fde_runtime_config_test.cpp ../fde_runtime_config.cpp ../fde_state.cpp
)

AddTest(
  TARGET MODULE_FDE_projected_hamiltonian
  SOURCES fde_projected_hamiltonian_test.cpp
    ../fde_projected_hamiltonian.cpp
)

AddTest(
  TARGET MODULE_FDE_density_artifact
  SOURCES fde_density_artifact_test.cpp ../fde_density_artifact.cpp
)

AddTest(
  TARGET MODULE_FDE_fragment_artifact
  SOURCES fde_fragment_artifact_test.cpp
    ../fde_fragment_artifact.cpp
    ../fde_ao_projection.cpp
    ../fde_determinant_artifact.cpp
)

AddTest(
  TARGET MODULE_FDE_determinant_artifact
  SOURCES fde_determinant_artifact_test.cpp
    ../fde_ao_projection.cpp
    ../fde_determinant_artifact.cpp
)

AddTest(
  TARGET MODULE_FDE_diabatic_assembler
  SOURCES fde_diabatic_assembler_test.cpp
    ../fde_ao_projection.cpp
    ../fde_determinant_artifact.cpp
    ../fde_diabatic_assembler.cpp
    ../fde_electronic_coupling.cpp
)

AddTest(
  TARGET MODULE_FDE_diabatic_postprocess
  SOURCES fde_diabatic_postprocess_test.cpp
    ../fde_ao_projection.cpp
    ../fde_determinant_artifact.cpp
    ../fde_diabatic_assembler.cpp
    ../fde_diabatic_postprocess.cpp
    ../fde_electronic_coupling.cpp
    ../fde_linearized_state.cpp
    ../fde_multistate_solver.cpp
    ../fde_runtime_config.cpp
    ../fde_state.cpp
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME MODULE_FDE_workflow_serial
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/fde/test_fde_workflow_e2e.py -v
)
set_tests_properties(MODULE_FDE_workflow_serial PROPERTIES
  ENVIRONMENT "OMP_NUM_THREADS=1"
)

if(ENABLE_MPI)
  add_executable(MODULE_FDE_electronic_coupling_mpi
    fde_electronic_coupling_mpi_test.cpp
    ../fde_ao_projection.cpp
    ../fde_determinant_artifact.cpp
    ../fde_electronic_coupling.cpp
  )
  target_link_libraries(MODULE_FDE_electronic_coupling_mpi PRIVATE
    MPI::MPI_CXX GTest::gtest_main GTest::gmock_main
  )
  install(TARGETS MODULE_FDE_electronic_coupling_mpi
    DESTINATION ${CMAKE_BINARY_DIR}/tests)
  add_test(
    NAME MODULE_FDE_electronic_coupling_2rank
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_FDE_electronic_coupling_mpi>
      ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(MODULE_FDE_electronic_coupling_2rank PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1"
    PROCESSORS 2
  )
endif()

AddTest(
  TARGET MODULE_FDE_electronic_coupling
  SOURCES fde_electronic_coupling_test.cpp
    ../fde_ao_projection.cpp
    ../fde_determinant_artifact.cpp
    ../fde_electronic_coupling.cpp
)

AddTest(
  TARGET MODULE_FDE_linearized_state
  SOURCES fde_linearized_state_test.cpp
    ../fde_ao_projection.cpp
    ../fde_determinant_artifact.cpp
    ../fde_electronic_coupling.cpp
    ../fde_linearized_state.cpp
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
  SOURCES fde_potential_test.cpp
    ../fde_libxc_pbe_provider.cpp
    ../fde_potential_evaluator.cpp
    ../fde_semilocal_functional.cpp
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

AddTest(
  TARGET MODULE_FDE_multi_fragment_freeze_thaw
  SOURCES fde_multi_fragment_freeze_thaw_test.cpp
    ../fde_density_artifact.cpp
    ../fde_multi_fragment_energy.cpp
    ../fde_multi_fragment_freeze_thaw.cpp
)

AddTest(
  TARGET MODULE_FDE_multistate_solver
  SOURCES fde_multistate_solver_test.cpp ../fde_multistate_solver.cpp
)

AddTest(
  TARGET MODULE_FDE_pes_scan
  SOURCES fde_pes_scan_test.cpp
    ../fde_pes_scan.cpp
    ../fde_density_artifact.cpp
    ../fde_energy_ledger.cpp
)

AddTest(
  TARGET MODULE_FDE_finite_difference
  SOURCES fde_finite_difference_test.cpp
    ../fde_finite_difference.cpp
    ../fde_density_artifact.cpp
    ../fde_energy_ledger.cpp
)
