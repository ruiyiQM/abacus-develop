#include "../fde_fragment_artifact.h"

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

namespace
{

fde::FragmentScfArtifact artifact()
{
    fde::FragmentScfArtifact value;
    value.schema_version = 1;
    value.state_label = "reactant";
    value.fragment_label = "F";
    value.geometry_fingerprint = "geom-000";
    value.orbital_fingerprint = "nao-v1";
    value.density_path = "reactant_F.fde_density";
    value.freeze_thaw_cycle = 3;
    value.scf_converged = true;
    value.ao_dimension = 2;
    value.active_orbitals = {0};
    value.alpha.coefficients = {1.0, 0.0};
    value.alpha.orbital_energies_ry = {-1.0};
    value.alpha.source_fragment_labels = {"F"};
    value.beta = value.alpha;
    value.ao_overlap = {1.0, 0.1, 0.1, 1.0};
    value.hamiltonian_alpha_ry = {-1.0, 0.2, 0.2, -0.5};
    value.hamiltonian_beta_ry = {-0.9, 0.3, 0.3, -0.4};
    value.subsystem_total_energy_ry = -20.0;
    value.ion_ion_energy_ry = 5.0;
    value.hartree_cross_energy_ry = 0.4;
    value.nonadditive_kinetic_energy_ry = 0.2;
    value.nonadditive_xc_energy_ry = -0.1;
    return value;
}

} // namespace

TEST(FdeFragmentArtifact, RoundTripsEveryRuntimeQuantity)
{
    std::ostringstream output;
    fde::FragmentScfArtifactIO::write(output, artifact(), 1.0e-12);
    std::istringstream input(output.str());
    const fde::FragmentScfArtifact copy
        = fde::FragmentScfArtifactIO::read(input, 1.0e-12);

    EXPECT_EQ(copy.state_label, "reactant");
    EXPECT_EQ(copy.fragment_label, "F");
    EXPECT_EQ(copy.alpha.coefficients, (std::vector<double>{1.0, 0.0}));
    EXPECT_EQ(copy.ao_overlap, (std::vector<double>{1.0, 0.1, 0.1, 1.0}));
    EXPECT_DOUBLE_EQ(copy.nonadditive_xc_energy_ry, -0.1);
}

TEST(FdeFragmentArtifact, RejectsAsymmetricHamiltonian)
{
    fde::FragmentScfArtifact value = artifact();
    value.hamiltonian_alpha_ry[1] = 0.5;
    EXPECT_THROW(fde::FragmentScfArtifactIO::validate(value, 1.0e-12),
                 std::invalid_argument);
}
