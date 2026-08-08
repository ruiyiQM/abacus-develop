set(FDE_LCAO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fde_ao_projection.cpp
    PARENT_SCOPE
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
