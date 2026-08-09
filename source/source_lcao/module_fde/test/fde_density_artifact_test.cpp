#include "../fde_density_artifact.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

namespace
{

fde::FrozenDensityArtifact artifact(const std::string& fragment_label)
{
    fde::FrozenDensityArtifact value;
    value.schema_version = 1;
    value.fragment_label = fragment_label;
    value.state_label = "state_1";
    value.geometry_fingerprint = "geometry_sha256";
    value.grid_fingerprint = "grid_sha256";
    value.pseudopotential_fingerprint = "pseudo_sha256";
    value.orbital_fingerprint = "orbital_sha256";
    value.core_density_fingerprint = fragment_label + "_core_sha256";
    value.xc_functional = "PBE";
    value.kinetic_functional = "PW91K";
    value.grid_x = 2;
    value.grid_y = 1;
    value.grid_z = 1;
    value.cell_volume_bohr3 = 2.0;
    value.alpha_electrons = 2;
    value.beta_electrons = 1;
    value.freeze_thaw_cycle = 3;
    value.scf_converged = true;
    value.orbital_kinetic_energy_ry = 1.25;
    value.nonlocal_pseudopotential_energy_ry = -0.25;
    value.rho_alpha_bohr3 = {1.0, 1.0};
    value.rho_beta_bohr3 = {0.5, 0.5};
    return value;
}

} // namespace

TEST(FdeDensityArtifact, RoundTripsDeterministically)
{
    const fde::FrozenDensityArtifact original = artifact("A");
    std::ostringstream output;
    fde::DensityArtifactIO::write(output, original);

    std::istringstream input(output.str());
    const fde::FrozenDensityArtifact restored = fde::DensityArtifactIO::read(input, 1.0e-12);
    std::ostringstream second_output;
    fde::DensityArtifactIO::write(second_output, restored);

    EXPECT_EQ(second_output.str(), output.str());
    EXPECT_EQ(restored.alpha_electrons, 2);
    EXPECT_EQ(restored.beta_electrons, 1);
    EXPECT_EQ(restored.rho_alpha_bohr3, original.rho_alpha_bohr3);
}

TEST(FdeDensityArtifact, ValidatesCompatibleFragmentPairs)
{
    const fde::FrozenDensityArtifact first = artifact("A");
    fde::FrozenDensityArtifact second = artifact("B");
    EXPECT_NO_THROW(fde::DensityArtifactIO::validate_compatible_pair(first, second, 1.0e-12));

    second.grid_fingerprint = "different_grid";
    EXPECT_THROW(fde::DensityArtifactIO::validate_compatible_pair(first, second, 1.0e-12),
                 std::invalid_argument);
}

TEST(FdeDensityArtifact, RejectsIncorrectElectronIntegrals)
{
    fde::FrozenDensityArtifact invalid = artifact("A");
    invalid.rho_beta_bohr3[0] = 0.25;
    EXPECT_THROW(fde::DensityArtifactIO::validate(invalid, 1.0e-12), std::invalid_argument);
}

TEST(FdeDensityArtifact, RoundTripsPartialScfDensityWithoutPromotingIt)
{
    fde::FrozenDensityArtifact partial = artifact("A");
    partial.scf_converged = false;
    std::ostringstream output;
    fde::DensityArtifactIO::write(output, partial);

    std::istringstream input(output.str());
    const fde::FrozenDensityArtifact restored
        = fde::DensityArtifactIO::read(input, 1.0e-12);
    EXPECT_FALSE(restored.scf_converged);
}

TEST(FdeDensityArtifact, ExpandsCompactUniformRuntimeSeed)
{
    std::istringstream input(R"(FDE_UNIFORM_DENSITY_SEED 1
FRAGMENT F
STATE reactant
GEOMETRY geometry
GRID_FINGERPRINT grid
PSEUDOPOTENTIALS pseudo
ORBITALS basis
CORE_DENSITY none
FUNCTIONALS pbe lc94
GRID 2 1 1 2
POPULATIONS 1 0
RHO_UNIFORM 0.5 0
END
)");
    const fde::FrozenDensityArtifact seed
        = fde::DensityArtifactIO::read_runtime(input, 1.0e-12);
    EXPECT_FALSE(seed.scf_converged);
    ASSERT_EQ(seed.rho_alpha_bohr3.size(), 2);
    EXPECT_DOUBLE_EQ(seed.rho_alpha_bohr3[0], 0.5);
    EXPECT_DOUBLE_EQ(seed.rho_alpha_bohr3[1], 0.5);
    EXPECT_DOUBLE_EQ(seed.rho_beta_bohr3[0], 0.0);
}
