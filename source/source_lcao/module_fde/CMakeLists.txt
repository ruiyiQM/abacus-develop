set(FDE_LCAO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fde_ao_projection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fde_density_artifact.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fde_semilocal_functional.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fde_state.cpp
    PARENT_SCOPE
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
