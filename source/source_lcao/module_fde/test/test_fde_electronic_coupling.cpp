#include "../coupling/fde_electronic_coupling.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace
{

using Complex = std::complex<double>;

std::vector<Complex> metric()
{
    return {1.0, 0.0, 0.0, 1.0};
}

fde::ComplexDiabaticDeterminantArtifact determinant(
    const std::string& label,
    const std::vector<Complex>& alpha)
{
    fde::ComplexDiabaticDeterminantArtifact result;
    result.schema_version = 2;
    result.state_label = label;
    result.geometry_fingerprint = "fixed_cell";
    result.orbital_fingerprint = "fixed_basis";
    result.ao_dimension = 2;
    result.boundary_twist = {0.17, 0.11, 0.0};
    result.alpha.coefficients = alpha;
    result.alpha.orbital_energies_ry = {-1.0};
    result.alpha.source_fragment_labels = {"D"};
    result.beta.coefficients = {0.0, 1.0};
    result.beta.orbital_energies_ry = {-2.0};
    result.beta.source_fragment_labels = {"A"};
    return result;
}

class OneBodyEnergy : public fde::ComplexTransitionEnergyProvider
{
  public:
    explicit OneBodyEnergy(const std::vector<Complex>& alpha_h) : alpha_h_(alpha_h) {}

    std::string name() const override
    {
        return "analytic_one_body";
    }

    Complex evaluate_ry(const fde::ComplexDiabaticDeterminantArtifact&,
                        const fde::ComplexDiabaticDeterminantArtifact&,
                        const fde::ComplexSpinTransitionDensityMatrix& density,
                        const std::size_t dimension) const override
    {
        const std::size_t matrix_size = dimension * dimension;
        EXPECT_EQ(alpha_h_.size(), matrix_size);
        Complex result = 0.0;
        for (std::size_t column = 0; column < dimension; ++column)
        {
            for (std::size_t row = 0; row < dimension; ++row)
            {
                result += density.alpha[row + column * dimension]
                          * alpha_h_[column + row * dimension];
            }
            result += 3.0 * density.beta[column + column * dimension];
        }
        return result;
    }

  private:
    const std::vector<Complex> alpha_h_;
};

void expect_complex_near(const Complex actual, const Complex expected)
{
    EXPECT_NEAR(actual.real(), expected.real(), 1.0e-12);
    EXPECT_NEAR(actual.imag(), expected.imag(), 1.0e-12);
}

void rotate_occupied(fde::ComplexOccupiedSpinOrbitals& orbitals,
                     const std::size_t dimension)
{
    const auto old = orbitals.coefficients;
    const Complex mixing(0.0, 0.6);
    for (std::size_t row = 0; row < dimension; ++row)
    {
        orbitals.coefficients[row] = 0.8 * old[row] + mixing * old[row + dimension];
        orbitals.coefficients[row + dimension] = mixing * old[row] + 0.8 * old[row + dimension];
    }
}

} // namespace

TEST(FdeComplexElectronicCoupling, MatchesIndependentComplexOneBodySpectrumAndLowdinCoupling)
{
    const auto first = determinant("D", {1.0, 0.0});
    const auto second = determinant("A", {0.8, Complex(0.0, 0.6)});
    const auto overlap = metric();
    const std::vector<Complex> alpha_h = {2.0, Complex(0.5, -0.25),
                                         Complex(0.5, 0.25), 1.0};
    const OneBodyEnergy energy(alpha_h);
    const fde::CouplingValidationControls controls{1.0e-12, 1.0e-12};
    const auto result = fde::ComplexElectronicCoupling::evaluate_symmetric(
        first, second, overlap, energy, controls, 1.0e-12);
    // H_alpha has eigenvalues 3/4 and 9/4; the beta spectator contributes 3 Ry.
    expect_complex_near(result.normalized_overlap, 0.8);
    const Complex expected_h(3.85, 0.3);
    expect_complex_near(result.hamiltonian_coupling_ry, expected_h);
    const auto pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
        5.0, 4.4, result.normalized_overlap, result.hamiltonian_coupling_ry, 1.0e-10);
    const Complex expected_v(0.25, 0.5);
    expect_complex_near(pair.coupling_ry, expected_v);
    EXPECT_NEAR(pair.eigenvalues_ry[0], 3.75, 1.0e-12);
    EXPECT_NEAR(pair.eigenvalues_ry[1], 5.25, 1.0e-12);
    const Complex conjugate_h = std::conj(pair.hamiltonian_ry[1]);
    expect_complex_near(pair.hamiltonian_ry[2], conjugate_h);
}

TEST(FdeComplexElectronicCoupling, PreservesTraceReciprocityAndOrbitalPhaseObservables)
{
    const auto first = determinant("D", {1.0, 0.0});
    auto second = determinant("A", {0.8, Complex(0.0, 0.6)});
    const auto overlap = metric();
    const std::vector<Complex> alpha_h = {2.0, Complex(0.5, -0.25),
                                         Complex(0.5, 0.25), 1.0};
    const OneBodyEnergy energy(alpha_h);
    const fde::CouplingValidationControls controls{1.0e-12, 1.0e-12};
    const auto forward = fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12);
    const auto reverse = fde::ComplexElectronicCoupling::transition(second, first, overlap, 1.0e-12);
    for (std::size_t column = 0; column < 2; ++column)
    {
        for (std::size_t row = 0; row < 2; ++row)
        {
            const Complex conjugate = std::conj(reverse.density_matrix.alpha[column + row * 2]);
            expect_complex_near(forward.density_matrix.alpha[row + column * 2], conjugate);
        }
    }
    const auto alpha_count = fde::ComplexElectronicCoupling::electron_count(
        forward.density_matrix.alpha, overlap, 2);
    const auto beta_count = fde::ComplexElectronicCoupling::electron_count(
        forward.density_matrix.beta, overlap, 2);
    expect_complex_near(alpha_count, 1.0);
    expect_complex_near(beta_count, 1.0);
    const Complex phase = std::polar(1.0, 0.7);
    for (auto& coefficient : second.alpha.coefficients)
    {
        coefficient *= phase;
    }
    const auto changed = fde::ComplexElectronicCoupling::evaluate_symmetric(
        first, second, overlap, energy, controls, 1.0e-12);
    const auto pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
        5.0, 4.4, changed.normalized_overlap, changed.hamiltonian_coupling_ry, 1.0e-10);
    const double magnitude = std::abs(pair.coupling_ry);
    const double expected = std::sqrt(5.0) / 4.0;
    EXPECT_NEAR(magnitude, expected, 1.0e-12);
    EXPECT_NEAR(pair.eigenvalues_ry[0], 3.75, 1.0e-12);
    EXPECT_NEAR(pair.eigenvalues_ry[1], 5.25, 1.0e-12);
    for (std::size_t index = 0; index < 4; ++index)
    {
        expect_complex_near(changed.forward_density_matrix.alpha[index], forward.density_matrix.alpha[index]);
    }
}

TEST(FdeComplexElectronicCoupling, IsInvariantUnderComplexOccupiedSubspaceRotation)
{
    auto first = determinant("D", {1.0, 0.0});
    auto second = determinant("A", {0.8, Complex(0.0, 0.6)});
    std::vector<Complex> overlap(16, 0.0);
    for (std::size_t index = 0; index < 4; ++index)
    {
        overlap[index + index * 4] = 1.0;
    }
    for (auto* state : {&first, &second})
    {
        state->ao_dimension = 4;
        state->alpha.coefficients.resize(8, 0.0);
        state->alpha.coefficients[6] = 1.0;
        state->alpha.orbital_energies_ry.push_back(-0.5);
        state->alpha.source_fragment_labels.push_back("D");
        state->beta.coefficients = {0.0, 0.0, 0.0, 1.0};
    }
    const auto reference = fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12);
    rotate_occupied(first.alpha, 4);
    rotate_occupied(second.alpha, 4);
    const auto changed = fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12);
    expect_complex_near(changed.normalized_overlap, reference.normalized_overlap);
    for (std::size_t index = 0; index < 16; ++index)
    {
        expect_complex_near(changed.density_matrix.alpha[index], reference.density_matrix.alpha[index]);
    }
}

TEST(FdeComplexElectronicCoupling, TransfersOneLocalElectronWithAnUnchangedSpectator)
{
    auto first = determinant("D", {1.0, 0.0});
    auto second = determinant("A", {0.0, 1.0});
    for (auto* state : {&first, &second})
    {
        state->ao_dimension = 3;
        state->alpha.coefficients.resize(3, 0.0);
        state->beta.coefficients = {0.0, 0.0, 1.0};
        state->beta.source_fragment_labels = {"E"};
    }
    second.alpha.source_fragment_labels = {"A"};
    // Local |D>, |A> have <D|A>=4i/5; |E> is an orthogonal beta spectator.
    const std::vector<Complex> overlap = {1.0, Complex(0.0, -0.8), 0.0,
                                         Complex(0.0, 0.8), 1.0, 0.0,
                                         0.0, 0.0, 1.0};
    const std::vector<Complex> alpha_h = {2.0, Complex(-0.3, -1.45), 0.0,
                                         Complex(-0.3, 1.45), 1.4, 0.0,
                                         0.0, 0.0, 0.0};
    const OneBodyEnergy energy(alpha_h);
    const fde::CouplingValidationControls controls{1.0e-12, 1.0e-12};
    const auto result = fde::ComplexElectronicCoupling::evaluate_symmetric(
        first, second, overlap, energy, controls, 1.0e-12);
    const Complex expected_s(0.0, 0.8);
    const Complex expected_h(-0.3, 3.85);
    expect_complex_near(result.normalized_overlap, expected_s);
    expect_complex_near(result.hamiltonian_coupling_ry, expected_h);
    const auto pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
        5.0, 4.4, result.normalized_overlap, result.hamiltonian_coupling_ry, 1.0e-10);
    const Complex expected_v(-0.5, 0.25);
    expect_complex_near(pair.coupling_ry, expected_v);
    EXPECT_NEAR(pair.eigenvalues_ry[0], 3.75, 1.0e-12);
    EXPECT_NEAR(pair.eigenvalues_ry[1], 5.25, 1.0e-12);
    const auto alpha_count = fde::ComplexElectronicCoupling::electron_count(
        result.forward_density_matrix.alpha, overlap, 3);
    const auto beta_count = fde::ComplexElectronicCoupling::electron_count(
        result.forward_density_matrix.beta, overlap, 3);
    expect_complex_near(alpha_count, 1.0);
    expect_complex_near(beta_count, 1.0);
    expect_complex_near(result.forward_density_matrix.beta[8], 1.0);
}

TEST(FdeComplexElectronicCoupling, RealGammaAndZeroOverlapLimitsAgree)
{
    const auto real_pair = fde::ElectronicCoupling::orthogonalize_pair(5.0, 5.12, 0.8, 4.3, 1.0e-10);
    const auto complex_pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
        5.0, 5.12, 0.8, 4.3, 1.0e-10);
    for (std::size_t index = 0; index < 4; ++index)
    {
        expect_complex_near(complex_pair.hamiltonian_ry[index], real_pair.hamiltonian_ry[index]);
    }
    EXPECT_NEAR(real_pair.coupling_ry, 0.7, 1.0e-12);
    const Complex imaginary_h(0.0, 0.5);
    const auto zero_pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
        1.0, 1.0, 0.0, imaginary_h, 1.0e-10);
    expect_complex_near(zero_pair.coupling_ry, imaginary_h);
    EXPECT_NEAR(zero_pair.eigenvalues_ry[0], 0.5, 1.0e-12);
    EXPECT_NEAR(zero_pair.eigenvalues_ry[1], 1.5, 1.0e-12);
}

TEST(FdeComplexElectronicCoupling, IsInvariantUnderComplexAoBasisTransformation)
{
    auto first = determinant("D", {1.0, 0.0});
    auto second = determinant("A", {0.8, Complex(0.0, 0.6)});
    const double scale = 1.0 / std::sqrt(2.0);
    const std::vector<Complex> unitary = {scale, Complex(0.0, scale),
                                         Complex(0.0, scale), scale};
    const std::vector<Complex> old_h = {2.0, Complex(0.5, -0.25),
                                       Complex(0.5, 0.25), 1.0};
    std::vector<Complex> transformed_h(4, 0.0);
    for (std::size_t column = 0; column < 2; ++column)
    {
        for (std::size_t row = 0; row < 2; ++row)
        {
            for (std::size_t left = 0; left < 2; ++left)
            {
                for (std::size_t right = 0; right < 2; ++right)
                {
                    transformed_h[row + column * 2] += std::conj(unitary[left + row * 2])
                        * old_h[left + right * 2] * unitary[right + column * 2];
                }
            }
        }
    }
    for (auto* state : {&first, &second})
    {
        for (auto* spin : {&state->alpha, &state->beta})
        {
            const auto old = spin->coefficients;
            for (std::size_t row = 0; row < 2; ++row)
            {
                spin->coefficients[row] = 0.0;
                for (std::size_t column = 0; column < 2; ++column)
                {
                    spin->coefficients[row] += std::conj(unitary[column + row * 2]) * old[column];
                }
            }
        }
    }
    const auto overlap = metric();
    const OneBodyEnergy energy(transformed_h);
    const fde::CouplingValidationControls controls{1.0e-12, 1.0e-12};
    const auto result = fde::ComplexElectronicCoupling::evaluate_symmetric(
        first, second, overlap, energy, controls, 1.0e-12);
    const Complex expected_h(3.85, 0.3);
    expect_complex_near(result.hamiltonian_coupling_ry, expected_h);
    expect_complex_near(result.normalized_overlap, 0.8);
}

TEST(FdeComplexElectronicCoupling, RejectsSingularOverlapAndBoundaryOrPopulationMixing)
{
    const auto first = determinant("D", {1.0, 0.0});
    auto second = determinant("A", {1.0e-14, Complex(0.0, 1.0)});
    const auto overlap = metric();
    EXPECT_THROW(fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12),
                 std::runtime_error);
    second = determinant("A", {0.8, Complex(0.0, 0.6)});
    second.boundary_twist[0] = -0.17;
    EXPECT_THROW(fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12),
                 std::invalid_argument);
    second.boundary_twist = first.boundary_twist;
    second.beta = fde::ComplexOccupiedSpinOrbitals();
    EXPECT_THROW(fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12),
                 std::invalid_argument);
    EXPECT_THROW(fde::ComplexElectronicCoupling::orthogonalize_pair(1.0, 1.0, 1.0, 0.5, 1.0e-10),
                 std::runtime_error);
}
