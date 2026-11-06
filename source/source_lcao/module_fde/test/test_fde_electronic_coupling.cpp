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
    result.resource_fingerprint = "fixed_pp_ao_grid_functionals";
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

template <typename Scalar>
void check_occupied_self_gram(const int schema_version,
                              const std::vector<double>& twist,
                              const Scalar phase)
{
    for (const std::size_t dimension : {2u, 4u})
    {
        fde::ScalarDiabaticDeterminantArtifact<Scalar> state;
        state.schema_version = schema_version;
        state.state_label = "self";
        state.geometry_fingerprint = "fixed_cell";
        state.orbital_fingerprint = "fixed_basis";
        state.resource_fingerprint = "fixed_resources";
        state.boundary_twist = twist;
        state.ao_dimension = dimension;
        const std::size_t matrix_size = dimension * dimension;
        state.alpha.coefficients.assign(matrix_size, 0.0);
        state.alpha.orbital_energies_ry.assign(dimension, -1.0);
        for (std::size_t orbital = 0; orbital < dimension; ++orbital)
        {
            state.alpha.coefficients[orbital + orbital * dimension] = 1.0;
            const std::string label = "fragment_" + std::to_string(orbital);
            state.alpha.source_fragment_labels.push_back(label);
        }
        state.beta = state.alpha;
        for (const double cross : {2.0, 0.25, 1.0 - 1.0e-14})
        {
            std::vector<Scalar> overlap(matrix_size, 0.0);
            for (std::size_t orbital = 0; orbital < dimension; ++orbital)
            {
                overlap[orbital + orbital * dimension] = 1.0;
            }
            for (std::size_t orbital = 0; orbital < dimension; orbital += 2)
            {
                const Scalar cross_overlap = cross * phase;
                overlap[orbital + (orbital + 1) * dimension] = cross_overlap;
                const Scalar conjugate = fde::conjugate_scalar(cross_overlap);
                overlap[orbital + 1 + orbital * dimension] = conjugate;
            }
            if (cross == 0.25)
            {
                const auto self = fde::ScalarElectronicCoupling<Scalar>::transition(
                    state, state, overlap, 1.0e-12);
                const Scalar norm_difference = self.normalized_overlap - 1.0;
                const double norm_error = std::abs(norm_difference);
                EXPECT_NEAR(norm_error, 0.0, 1.0e-12);
                const Scalar count = fde::ScalarElectronicCoupling<Scalar>::electron_count(
                    self.density_matrix.alpha, overlap, dimension);
                const Scalar trace_difference = count - static_cast<double>(dimension);
                const double trace_error = std::abs(trace_difference);
                EXPECT_NEAR(trace_error, 0.0, 1.0e-12);
                auto empty_beta = state;
                empty_beta.beta = fde::ScalarOccupiedSpinOrbitals<Scalar>();
                EXPECT_NO_THROW(fde::ScalarElectronicCoupling<Scalar>::transition(
                    empty_beta, empty_beta, overlap, 1.0e-12));
            }
            else
            {
                // At cross=2 each block has eigenvalues 3,-1, even when its determinant product is positive.
                EXPECT_THROW(fde::ScalarElectronicCoupling<Scalar>::transition(
                    state, state, overlap, 1.0e-12), std::runtime_error);
                auto empty_alpha = state;
                empty_alpha.alpha = fde::ScalarOccupiedSpinOrbitals<Scalar>();
                EXPECT_THROW(fde::ScalarElectronicCoupling<Scalar>::transition(
                    empty_alpha, empty_alpha, overlap, 1.0e-12), std::runtime_error);
            }
        }
    }
}

} // namespace

TEST(FdeElectronicCoupling, ValidatesEveryOccupiedSelfGram)
{
    const std::vector<double> twist;
    check_occupied_self_gram<double>(1, twist, 1.0);
}

TEST(FdeComplexElectronicCoupling, ValidatesEveryOccupiedSelfGram)
{
    const std::vector<double> twist = {0.17, 0.11, 0.0};
    const Complex phase(0.0, 1.0);
    check_occupied_self_gram<Complex>(2, twist, phase);
}

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
    EXPECT_LE(pair.maximum_residual, 1.0e-12);
    EXPECT_LE(pair.maximum_metric_error, 1.0e-12);
    ASSERT_EQ(pair.coefficients.size(), 4);
    const Complex reversed_h = std::conj(expected_h);
    for (std::size_t root = 0; root < 2; ++root)
    {
        const Complex c0 = pair.coefficients[root * 2];
        const Complex c1 = pair.coefficients[1 + root * 2];
        const Complex sc0 = c0 + 0.8 * c1;
        const Complex sc1 = 0.8 * c0 + c1;
        const Complex hc0 = 5.0 * c0 + expected_h * c1;
        const Complex hc1 = reversed_h * c0 + 4.4 * c1;
        const Complex right0 = pair.eigenvalues_ry[root] * sc0;
        const Complex right1 = pair.eigenvalues_ry[root] * sc1;
        const Complex norm = std::conj(c0) * sc0 + std::conj(c1) * sc1;
        expect_complex_near(hc0, right0);
        expect_complex_near(hc1, right1);
        expect_complex_near(norm, 1.0);
    }
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

TEST(FdeElectronicCoupling, PreservesSmallCouplingAcrossLargeEnergyZeroShifts)
{
    const double large_shift = std::ldexp(1.0, 40);
    const double gap = std::sqrt(39.0) / 12.0;
    for (const double shift : {0.0, large_shift, -large_shift})
    {
        const double first = shift + 0.125;
        const double second = shift - 0.125;
        const double h12 = 0.5 * shift + 0.375;
        const auto pair = fde::ElectronicCoupling::orthogonalize_pair(
            first, second, 0.5, h12, 1.0e-10);
        // det(H_res-e*S)=3*e^2/4+3*e/8-5/32; V=3/8/(1-1/4).
        const double lower = shift - 0.25 - gap;
        const double upper = shift - 0.25 + gap;
        const double root_tolerance = shift == 0.0 ? 1.0e-12 : 0.000244140625;
        EXPECT_NEAR(pair.coupling_ry, 0.5, 1.0e-12);
        EXPECT_NEAR(pair.eigenvalues_ry[0], lower, root_tolerance);
        EXPECT_NEAR(pair.eigenvalues_ry[1], upper, root_tolerance);
        EXPECT_LE(pair.maximum_residual, 1.0e-12);
        EXPECT_LE(pair.maximum_metric_error, 1.0e-12);
    }
}

TEST(FdeComplexElectronicCoupling, PreservesSmallComplexCouplingAcrossLargeEnergyZeroShifts)
{
    const double large_shift = std::ldexp(1.0, 40);
    const double gap = std::sqrt(46.0) / 12.0;
    const Complex overlap(0.0, 0.5);
    const double expected_real = std::sqrt(3.0) / 4.0;
    const Complex expected_v(expected_real, 1.0 / 3.0);
    for (const double shift : {0.0, large_shift, -large_shift})
    {
        const double first = shift + 0.125;
        const double second = shift - 0.125;
        const double imaginary = 0.5 * shift + 0.25;
        const Complex h12(0.375, imaginary);
        const auto pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
            first, second, overlap, h12, 1.0e-10);
        // det(H_res-e*S)=3*e^2/4+e/4-7/32; roots=(-2 +/- sqrt(46))/12.
        const double lower = shift - 1.0 / 6.0 - gap;
        const double upper = shift - 1.0 / 6.0 + gap;
        const double root_tolerance = shift == 0.0 ? 1.0e-12 : 0.000244140625;
        expect_complex_near(pair.coupling_ry, expected_v);
        EXPECT_NEAR(pair.eigenvalues_ry[0], lower, root_tolerance);
        EXPECT_NEAR(pair.eigenvalues_ry[1], upper, root_tolerance);
        EXPECT_LE(pair.maximum_residual, 1.0e-12);
        EXPECT_LE(pair.maximum_metric_error, 1.0e-12);
    }
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
    second.resource_fingerprint = "different_pseudopotential";
    EXPECT_THROW(fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12),
                 std::invalid_argument);
    second.resource_fingerprint = first.resource_fingerprint;
    second.beta = fde::ComplexOccupiedSpinOrbitals();
    EXPECT_THROW(fde::ComplexElectronicCoupling::transition(first, second, overlap, 1.0e-12),
                 std::invalid_argument);
    EXPECT_THROW(fde::ComplexElectronicCoupling::orthogonalize_pair(1.0, 1.0, 1.0, 0.5, 1.0e-10),
                 std::runtime_error);
}

TEST(FdeComplexElectronicCoupling, HandlesDiagonalAndExactlyDegenerateRootVectors)
{
    for (const double second_energy : {1.0, 2.0, 0.0})
    {
        const auto pair = fde::ComplexElectronicCoupling::orthogonalize_pair(
            1.0, second_energy, 0.0, 0.0, 1.0e-10);
        EXPECT_LE(pair.maximum_residual, 1.0e-12);
        EXPECT_LE(pair.maximum_metric_error, 1.0e-12);
        ASSERT_EQ(pair.coefficients.size(), 4);
    }
}
