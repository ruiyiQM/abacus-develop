#include "../runtime/fde_session_contract.h"
#include "../fde_runtime_config.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

namespace
{

fde::FdeRuntimeConfig config()
{
    std::istringstream input(R"(FDE_CONFIG 1
ATOM_COUNT 2
FRAGMENT A 2 1 0
FRAGMENT B 2 1 1
STATE s 0 0 2 A 0 0 B 0 0
ACTIVE_STATE s
ACTIVE_FRAGMENT A
ACTIVE_DENSITY /tmp/a0
FROZEN_DENSITY B /tmp/b0
OUTPUT_PREFIX /tmp/result0
KEDF pw91k
DENSITY_FLOOR_BOHR3 1e-12
MAX_SCF_ITERATIONS 20
SCF_DENSITY_TOLERANCE 1e-6
ELECTRON_TOLERANCE 1e-8
MIXING_BETA 0.2
MAX_FREEZE_THAW_CYCLES 10
FREEZE_THAW_DENSITY_TOLERANCE 1e-5
ENERGY_TOLERANCE_RY 1e-6
UPDATE_ORDER 2 A B
K_STATES 0
L_FRAGMENTS 0
M_FRAGMENTS 0
SINGULAR_VALUE_TOLERANCE 1e-10
OVERLAP_EIGENVALUE_CUTOFF 1e-9
SYMMETRY_TOLERANCE 1e-10
RESIDUAL_TOLERANCE 1e-8
MINIMUM_ROOT_OVERLAP 0.5
CALCULATE_FORCE false
HARTREE_RECIPROCITY_TOLERANCE_RY 1e-8
END_FDE_CONFIG
)");
    return fde::FdeRuntimeConfigIO::read(input);
}

TEST(FdeSessionContract, AllowsMutableDensityPathsAndOutput)
{
    const fde::FdeRuntimeConfig initialized = config();
    fde::FdeRuntimeConfig requested = initialized;
    requested.active_density_path = "/tmp/a1";
    requested.frozen_density_artifacts[0].path = "/tmp/b1";
    requested.output_prefix = "/tmp/result1";
    EXPECT_NO_THROW(
        fde::FdeSessionContract::validate_compatible(initialized, requested));
}

TEST(FdeSessionContract, RejectsChangedActiveFragment)
{
    const fde::FdeRuntimeConfig initialized = config();
    fde::FdeRuntimeConfig requested = initialized;
    requested.active_fragment = "B";
    EXPECT_THROW(
        fde::FdeSessionContract::validate_compatible(initialized, requested),
        std::invalid_argument);
}

TEST(FdeSessionContract, RejectsChangedScfControls)
{
    const fde::FdeRuntimeConfig initialized = config();
    fde::FdeRuntimeConfig requested = initialized;
    requested.scf_density_tolerance = 1.0e-8;
    EXPECT_THROW(
        fde::FdeSessionContract::validate_compatible(initialized, requested),
        std::invalid_argument);
}

} // namespace
