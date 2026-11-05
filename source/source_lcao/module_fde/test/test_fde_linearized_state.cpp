#include "../coupling/fde_linearized_state.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

namespace
{

using Complex = std::complex<double>;

fde::ComplexDiabaticDeterminantArtifact determinant(const std::string& label,
                                                   const std::vector<Complex>& alpha)
{
    fde::ComplexDiabaticDeterminantArtifact value;
    value.schema_version = 2;
    value.state_label = label;
    value.geometry_fingerprint = "fixed_cell";
    value.orbital_fingerprint = "fixed_basis";
    value.boundary_twist = {0.17, 0.11, 0.0};
    value.ao_dimension = 2;
    value.alpha.coefficients = alpha;
    value.alpha.orbital_energies_ry = {-1.0};
    value.alpha.source_fragment_labels = {"D"};
    value.beta.coefficients = {0.0, 1.0};
    value.beta.orbital_energies_ry = {-2.0};
    value.beta.source_fragment_labels = {"A"};
    return value;
}

fde::ComplexLinearizedStateArtifact state(const std::string& label, const double energy)
{
    fde::ComplexLinearizedStateArtifact value;
    value.schema_version = 2;
    value.state_label = label;
    value.geometry_fingerprint = "fixed_cell";
    value.orbital_fingerprint = "fixed_basis";
    value.boundary_twist = {0.17, 0.11, 0.0};
    value.ao_dimension = 2;
    value.reference_energy_ry = energy;
    value.hamiltonian_alpha_ry = {2.0, Complex(0.5, -0.25), Complex(0.5, 0.25), 1.0};
    value.hamiltonian_beta_ry = {3.0, 0.0, 0.0, 3.0};
    return value;
}

} // namespace

TEST(FdeComplexLinearizedState, RoundTripsReferenceHamiltonianAndBoundaryIdentity)
{
    const auto value = state("D", 5.0);
    std::ostringstream output;
    fde::ComplexLinearizedStateArtifactIO::write(output, value, 1.0e-12);
    std::istringstream input(output.str());
    const auto restored = fde::ComplexLinearizedStateArtifactIO::read(input, 1.0e-12);
    EXPECT_EQ(restored.hamiltonian_alpha_ry, value.hamiltonian_alpha_ry);
    EXPECT_EQ(restored.boundary_twist, value.boundary_twist);
    EXPECT_EQ(restored.orbital_fingerprint, value.orbital_fingerprint);
    EXPECT_EQ(restored.reference_energy_ry, value.reference_energy_ry);
    std::istringstream wrong_scalar(output.str());
    EXPECT_THROW(fde::LinearizedStateArtifactIO::read(wrong_scalar, 1.0e-12), std::invalid_argument);
}

TEST(FdeComplexLinearizedState, AnchorsSelfEnergyAndRecoversIndependentOneBodyReference)
{
    const auto first = determinant("D", {1.0, 0.0});
    const auto second = determinant("A", {0.8, Complex(0.0, 0.6)});
    const std::vector<Complex> overlap = {1.0, 0.0, 0.0, 1.0};
    const std::vector<fde::ComplexLinearizedStateArtifact> states = {state("D", 5.0), state("A", 4.4)};
    const fde::ComplexLinearizedTransitionEnergy provider(states, overlap, 1.0e-12);
    const auto self = fde::ComplexElectronicCoupling::transition(first, first, overlap, 1.0e-12);
    const auto self_energy = provider.evaluate_ry(first, first, self.density_matrix, 2);
    EXPECT_NEAR(self_energy.real(), 5.0, 1.0e-12);
    EXPECT_NEAR(self_energy.imag(), 0.0, 1.0e-12);
    const fde::CouplingValidationControls controls{1.0e-12, 1.0e-12};
    const auto result = fde::ComplexElectronicCoupling::evaluate_symmetric(
        first, second, overlap, provider, controls, 1.0e-12);
    EXPECT_NEAR(result.hamiltonian_coupling_ry.real(), 3.85, 1.0e-12);
    EXPECT_NEAR(result.hamiltonian_coupling_ry.imag(), 0.3, 1.0e-12);
    const auto pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
        5.0, 4.4, result.normalized_overlap, result.hamiltonian_coupling_ry, 1.0e-10);
    EXPECT_NEAR(pair.eigenvalues_ry[0], 3.75, 1.0e-12);
    EXPECT_NEAR(pair.eigenvalues_ry[1], 5.25, 1.0e-12);
}

TEST(FdeComplexLinearizedState, RejectsMixedBoundaryBasisAndNonHermitianArtifact)
{
    const std::vector<Complex> overlap = {1.0, 0.0, 0.0, 1.0};
    std::vector<fde::ComplexLinearizedStateArtifact> states = {state("D", 5.0), state("A", 4.4)};
    states[1].boundary_twist[0] = -0.17;
    EXPECT_THROW(fde::ComplexLinearizedTransitionEnergy provider(states, overlap, 1.0e-12),
                 std::invalid_argument);
    states[1] = state("A", 4.4);
    states[1].orbital_fingerprint = "different_basis";
    EXPECT_THROW(fde::ComplexLinearizedTransitionEnergy provider(states, overlap, 1.0e-12),
                 std::invalid_argument);
    states[1] = state("A", 4.4);
    states[1].hamiltonian_alpha_ry[1] = states[1].hamiltonian_alpha_ry[2];
    EXPECT_THROW(fde::ComplexLinearizedStateArtifactIO::validate(states[1], 1.0e-12),
                 std::invalid_argument);
}
