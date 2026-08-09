#include "../fde_linearized_state.h"

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

namespace
{

fde::DiabaticDeterminantArtifact determinant(const std::string& label,
                                             const std::vector<double>& alpha)
{
    fde::DiabaticDeterminantArtifact result;
    result.schema_version = 1;
    result.state_label = label;
    result.geometry_fingerprint = "geometry";
    result.orbital_fingerprint = "basis";
    result.ao_dimension = 2;
    result.alpha.coefficients = alpha;
    result.alpha.orbital_energies_ry = {-1.0};
    result.alpha.source_fragment_labels = {"F"};
    result.beta.coefficients = {0.0, 1.0};
    result.beta.orbital_energies_ry = {-0.5};
    result.beta.source_fragment_labels = {"CH3Cl"};
    return result;
}

fde::LinearizedStateArtifact state(const std::string& label, const double energy)
{
    fde::LinearizedStateArtifact result;
    result.schema_version = 1;
    result.state_label = label;
    result.geometry_fingerprint = "geometry";
    result.ao_dimension = 2;
    result.reference_energy_ry = energy;
    result.hamiltonian_alpha_ry = {2.0, 0.5, 0.5, 1.0};
    result.hamiltonian_beta_ry = {1.0, 0.0, 0.0, 3.0};
    return result;
}

} // namespace

TEST(FdeLinearizedState, RoundTripsArtifact)
{
    std::ostringstream output;
    fde::LinearizedStateArtifactIO::write(output, state("first", -10.0), 1.0e-12);
    std::istringstream input(output.str());
    const fde::LinearizedStateArtifact copy
        = fde::LinearizedStateArtifactIO::read(input, 1.0e-12);
    EXPECT_EQ(copy.state_label, "first");
    EXPECT_DOUBLE_EQ(copy.reference_energy_ry, -10.0);
    EXPECT_DOUBLE_EQ(copy.hamiltonian_alpha_ry[1], 0.5);
}

TEST(FdeLinearizedState, ReproducesReferenceAndDirectionalFirstOrderEnergy)
{
    const std::vector<double> overlap = {1.0, 0.0, 0.0, 1.0};
    const fde::DiabaticDeterminantArtifact first
        = determinant("first", {1.0, 0.0});
    const fde::DiabaticDeterminantArtifact second
        = determinant("second", {0.8, 0.6});
    const fde::LinearizedTransitionEnergy energy(
        {state("first", -10.0), state("second", -9.0)}, overlap, 1.0e-12);
    const fde::DeterminantTransition self
        = fde::ElectronicCoupling::transition(first, first, overlap, 1.0e-12);
    EXPECT_NEAR(energy.evaluate_ry(first, first, self.density_matrix, 2),
                -10.0,
                1.0e-12);
    const fde::DeterminantTransition forward
        = fde::ElectronicCoupling::transition(first, second, overlap, 1.0e-12);
    EXPECT_NE(energy.evaluate_ry(first, second, forward.density_matrix, 2),
              -10.0);
}
