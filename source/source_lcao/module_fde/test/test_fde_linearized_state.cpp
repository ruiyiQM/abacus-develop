#include "../coupling/fde_linearized_state.h"

#include <gtest/gtest.h>

#include <limits>
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
    value.resource_fingerprint = "fixed_pp_ao_grid_functionals";
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
    value.resource_fingerprint = "fixed_pp_ao_grid_functionals";
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
    EXPECT_EQ(restored.resource_fingerprint, value.resource_fingerprint);
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

TEST(FdeComplexLinearizedState, DistinguishesDeterminantSelfFromSummedFragmentDensity)
{
    const std::vector<Complex> overlap = {1.0, Complex(0.0, -0.5), Complex(0.0, 0.5), 1.0};
    const std::vector<Complex> expected_self = {4.0 / 3.0, Complex(0.0, 2.0 / 3.0),
                                               Complex(0.0, -2.0 / 3.0), 4.0 / 3.0};
    std::vector<fde::ComplexLinearizedStateArtifact> states = {state("D", 3.0), state("A", 3.0)};
    for (auto& value : states)
    {
        value.hamiltonian_alpha_ry = {2.0, 0.0, 0.0, 1.0};
        value.hamiltonian_beta_ry.assign(4, 0.0);
    }
    const fde::ComplexLinearizedTransitionEnergy provider(states, overlap, 1.0e-12);
    for (int phased = 0; phased < 2; ++phased)
    {
        auto full = determinant("D", {1.0, 0.0});
        full.alpha.coefficients = {1.0, 0.0, 0.0, 1.0};
        if (phased == 1)
        {
            full.alpha.coefficients[3] = Complex(0.0, 1.0);
        }
        full.alpha.orbital_energies_ry = {-1.0, -0.5};
        full.alpha.source_fragment_labels = {"D", "A"};
        full.beta = fde::ComplexOccupiedSpinOrbitals();
        const auto self = fde::ComplexElectronicCoupling::transition(full, full, overlap, 1.0e-12);
        EXPECT_NEAR(self.raw_overlap.real(), 0.75, 1.0e-12);
        EXPECT_NEAR(self.normalized_overlap.real(), 1.0, 1.0e-12);
        for (std::size_t index = 0; index < expected_self.size(); ++index)
        {
            EXPECT_NEAR(self.density_matrix.alpha[index].real(), expected_self[index].real(), 1.0e-12);
            EXPECT_NEAR(self.density_matrix.alpha[index].imag(), expected_self[index].imag(), 1.0e-12);
        }
        const auto count = fde::ComplexElectronicCoupling::electron_count(self.density_matrix.alpha, overlap, 2);
        EXPECT_NEAR(count.real(), 2.0, 1.0e-12);
        EXPECT_NEAR(count.imag(), 0.0, 1.0e-12);
        // Psum=I also has Tr(S Psum)=2, but h=diag(2,1) gives 3 Ry instead of 4 Ry.
        const Complex determinant_one_body = 2.0 * self.density_matrix.alpha[0] + self.density_matrix.alpha[3];
        const auto anchored = provider.evaluate_ry(full, full, self.density_matrix, 2);
        EXPECT_NEAR(determinant_one_body.real(), 4.0, 1.0e-12);
        EXPECT_NEAR(anchored.real(), 3.0, 1.0e-12);
        EXPECT_NEAR(anchored.imag(), 0.0, 1.0e-12);
    }
}

TEST(FdeComplexLinearizedState, NormalizesBothNonorthogonalSpinDeterminants)
{
    auto first = determinant("D", {1.0, 0.0});
    first.alpha.coefficients = {1.0, 0.0, 0.0, 1.0};
    first.alpha.orbital_energies_ry = {-1.0, -0.5};
    first.alpha.source_fragment_labels = {"D", "A"};
    first.beta = first.alpha;
    auto second = first;
    second.state_label = "A";
    second.alpha.coefficients[0] = Complex(0.0, 1.0);
    second.beta.coefficients[0] = Complex(0.0, 1.0);
    const std::vector<Complex> overlap = {1.0, Complex(0.0, -0.5), Complex(0.0, 0.5), 1.0};
    const auto self = fde::ComplexElectronicCoupling::transition(first, first, overlap, 1.0e-12);
    const auto forward = fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12);
    // Each spin has det(C^dagger S C)=3/4; the two ket phases multiply to -1.
    EXPECT_NEAR(self.raw_overlap.real(), 9.0 / 16.0, 1.0e-12);
    EXPECT_NEAR(self.normalized_overlap.real(), 1.0, 1.0e-12);
    EXPECT_NEAR(forward.raw_overlap.real(), -9.0 / 16.0, 1.0e-12);
    EXPECT_NEAR(forward.normalized_overlap.real(), -1.0, 1.0e-12);
    EXPECT_NEAR(forward.normalized_overlap.imag(), 0.0, 1.0e-12);
    for (const auto* density : {&forward.density_matrix.alpha, &forward.density_matrix.beta})
    {
        const auto count = fde::ComplexElectronicCoupling::electron_count(*density, overlap, 2);
        EXPECT_NEAR(count.real(), 2.0, 1.0e-12);
        EXPECT_NEAR(count.imag(), 0.0, 1.0e-12);
    }
}

TEST(FdeComplexLinearizedState, MatchesNonorthogonalOneBodyMatrixElementsAndEnergyZeroShifts)
{
    auto first = determinant("D", {1.0, 0.0});
    auto second = determinant("A", {0.0, 1.0});
    for (auto* value : {&first, &second})
    {
        value->ao_dimension = 3;
        value->alpha.coefficients.resize(3, 0.0);
        value->beta.coefficients = {0.0, 0.0, 1.0};
        value->beta.source_fragment_labels = {"E"};
    }
    second.alpha.source_fragment_labels = {"A"};
    const std::vector<Complex> overlap = {1.0, Complex(0.0, -0.8), 0.0,
                                         Complex(0.0, 0.8), 1.0, 0.0,
                                         0.0, 0.0, 1.0};
    const std::vector<Complex> alpha_h = {2.0, Complex(-0.3, -1.45), 0.0,
                                         Complex(-0.3, 1.45), 1.4, 0.0,
                                         0.0, 0.0, 0.0};
    const fde::CouplingValidationControls controls{1.0e-12, 1.0e-12};
    for (int mode = 0; mode < 3; ++mode)
    {
        double energy_shift = 0.0;
        double alpha_shift = 0.0;
        double beta_shift = 0.0;
        if (mode == 1)
        {
            energy_shift = 1.25;
        }
        else if (mode == 2)
        {
            alpha_shift = 0.75;
            beta_shift = -0.25;
            energy_shift = alpha_shift + beta_shift;
        }
        std::vector<fde::ComplexLinearizedStateArtifact> states = {state("D", 5.0), state("A", 4.4)};
        for (auto& value : states)
        {
            value.ao_dimension = 3;
            value.reference_energy_ry += energy_shift;
            value.hamiltonian_alpha_ry = alpha_h;
            value.hamiltonian_beta_ry.resize(9);
            for (std::size_t index = 0; index < overlap.size(); ++index)
            {
                value.hamiltonian_alpha_ry[index] += alpha_shift * overlap[index];
                value.hamiltonian_beta_ry[index] = (3.0 + beta_shift) * overlap[index];
            }
        }
        const fde::ComplexLinearizedTransitionEnergy provider(states, overlap, 1.0e-12);
        const auto forward = fde::ComplexElectronicCoupling::evaluate_symmetric(
            first, second, overlap, provider, controls, 1.0e-12);
        const auto reverse = fde::ComplexElectronicCoupling::evaluate_symmetric(
            second, first, overlap, provider, controls, 1.0e-12);
        // Direct spin-product matrix element: h_DA + 3 S_DA = -0.3 + 3.85i.
        const Complex expected_s(0.0, 0.8);
        const Complex expected_h = Complex(-0.3, 3.85) + energy_shift * expected_s;
        const double expected_lower = 3.75 + energy_shift;
        const double expected_upper = 5.25 + energy_shift;
        const auto pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
            states[0].reference_energy_ry, states[1].reference_energy_ry,
            forward.normalized_overlap, forward.hamiltonian_coupling_ry, 1.0e-10);
        EXPECT_NEAR(forward.normalized_overlap.real(), 0.0, 1.0e-12);
        EXPECT_NEAR(forward.normalized_overlap.imag(), 0.8, 1.0e-12);
        EXPECT_NEAR(forward.hamiltonian_coupling_ry.real(), expected_h.real(), 1.0e-12);
        EXPECT_NEAR(forward.hamiltonian_coupling_ry.imag(), expected_h.imag(), 1.0e-12);
        EXPECT_NEAR(reverse.hamiltonian_coupling_ry.real(), expected_h.real(), 1.0e-12);
        EXPECT_NEAR(reverse.hamiltonian_coupling_ry.imag(), -expected_h.imag(), 1.0e-12);
        EXPECT_NEAR(pair.eigenvalues_ry[0], expected_lower, 1.0e-12);
        EXPECT_NEAR(pair.eigenvalues_ry[1], expected_upper, 1.0e-12);
        EXPECT_NEAR(pair.coupling_ry.real(), -0.5, 1.0e-12);
        EXPECT_NEAR(pair.coupling_ry.imag(), 0.25, 1.0e-12);
        for (const auto* density : {&forward.forward_density_matrix.alpha, &forward.forward_density_matrix.beta})
        {
            const auto count = fde::ComplexElectronicCoupling::electron_count(*density, overlap, 3);
            EXPECT_NEAR(count.real(), 1.0, 1.0e-12);
            EXPECT_NEAR(count.imag(), 0.0, 1.0e-12);
        }
        auto phased = second;
        for (auto& coefficient : phased.alpha.coefficients)
        {
            coefficient *= Complex(0.0, 1.0);
        }
        const auto changed = fde::ComplexElectronicCoupling::evaluate_symmetric(
            first, phased, overlap, provider, controls, 1.0e-12);
        const Complex expected_phased_h = Complex(0.0, 1.0) * expected_h;
        const auto phased_pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
            states[0].reference_energy_ry, states[1].reference_energy_ry,
            changed.normalized_overlap, changed.hamiltonian_coupling_ry, 1.0e-10);
        EXPECT_NEAR(changed.normalized_overlap.real(), -0.8, 1.0e-12);
        EXPECT_NEAR(changed.normalized_overlap.imag(), 0.0, 1.0e-12);
        EXPECT_NEAR(changed.hamiltonian_coupling_ry.real(), expected_phased_h.real(), 1.0e-12);
        EXPECT_NEAR(changed.hamiltonian_coupling_ry.imag(), expected_phased_h.imag(), 1.0e-12);
        EXPECT_NEAR(phased_pair.coupling_ry.real(), -0.25, 1.0e-12);
        EXPECT_NEAR(phased_pair.coupling_ry.imag(), -0.5, 1.0e-12);
        EXPECT_NEAR(phased_pair.eigenvalues_ry[0], expected_lower, 1.0e-12);
        EXPECT_NEAR(phased_pair.eigenvalues_ry[1], expected_upper, 1.0e-12);
    }
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
    states[1].resource_fingerprint = "different_pp_grid";
    EXPECT_THROW(fde::ComplexLinearizedTransitionEnergy provider(states, overlap, 1.0e-12),
                 std::invalid_argument);
    states[1] = state("A", 4.4);
    states[1].hamiltonian_alpha_ry[1] = states[1].hamiltonian_alpha_ry[2];
    EXPECT_THROW(fde::ComplexLinearizedStateArtifactIO::validate(states[1], 1.0e-12),
                 std::invalid_argument);
}

TEST(FdeLinearizedStateArtifactIO, RejectsOverflowingAoDimensionsBeforeMatrixArithmetic)
{
    const int half_width = (std::numeric_limits<std::size_t>::digits + 1) / 2;
    const std::size_t dimension = std::size_t(1) << half_width;
    for (int version = 1; version <= 2; ++version)
    {
        std::ostringstream text;
        text << "FDE_LINEARIZED_STATE " << version << "\nSTATE D\nGEOMETRY fixed_cell\n";
        if (version == 2)
        {
            text << "ORBITALS fixed_basis\nBOUNDARY_TWIST 0.17 0.11 0\nRESOURCES fixed_resources\n";
        }
        text << "AO_DIMENSION " << dimension << "\nREFERENCE_ENERGY_RY 1\n"
             << "HAMILTONIAN_ALPHA_RY 0\nHAMILTONIAN_BETA_RY 0\nEND\n";
        std::istringstream input(text.str());
        if (version == 1)
        {
            EXPECT_THROW(fde::LinearizedStateArtifactIO::read(input, 1.0e-12), std::invalid_argument);
        }
        else
        {
            EXPECT_THROW(fde::ComplexLinearizedStateArtifactIO::read(input, 1.0e-12), std::invalid_argument);
        }
    }
    auto invalid = state("D", 1.0);
    invalid.ao_dimension = dimension;
    invalid.hamiltonian_alpha_ry.clear();
    invalid.hamiltonian_beta_ry.clear();
    EXPECT_THROW(fde::ComplexLinearizedStateArtifactIO::validate(invalid, 1.0e-12), std::invalid_argument);
    std::ostringstream output;
    EXPECT_THROW(fde::ComplexLinearizedStateArtifactIO::write(output, invalid, 1.0e-12), std::invalid_argument);
    auto second_invalid = invalid;
    second_invalid.state_label = "A";
    const std::vector<fde::ComplexLinearizedStateArtifact> states = {invalid, second_invalid};
    const std::vector<Complex> empty_overlap;
    EXPECT_THROW(fde::ComplexLinearizedTransitionEnergy provider(states, empty_overlap, 1.0e-12),
                 std::invalid_argument);
}
