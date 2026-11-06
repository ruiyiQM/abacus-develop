#include "../coupling/fde_transition_energy_components.h"

#include <gtest/gtest.h>

#include <complex>
#include <limits>
#include <stdexcept>

namespace
{

using Complex = std::complex<double>;

void expect_complex_near(const Complex actual, const Complex expected)
{
    EXPECT_NEAR(actual.real(), expected.real(), 1.0e-12);
    EXPECT_NEAR(actual.imag(), expected.imag(), 1.0e-12);
}

} // namespace

TEST(FdeTransitionEnergyComponents, ContractsRealNonsymmetricDensityWithoutAnExtraHalf)
{
    const std::vector<double> density = {1.0, 0.25, -0.5, 0.5};
    const std::vector<double> h = {2.0, 3.0, 3.0, 4.0};
    const double energy = fde::one_body_energy_ry(density, h, 2);
    EXPECT_NEAR(energy, 3.25, 1.0e-12);
    const std::vector<Complex> complex_density = {1.0, 0.25, -0.5, 0.5};
    const std::vector<Complex> complex_h = {2.0, 3.0, 3.0, 4.0};
    const Complex gamma_energy = fde::one_body_energy_ry(complex_density, complex_h, 2);
    expect_complex_near(gamma_energy, energy);
}

TEST(FdeTransitionEnergyComponents, KeepsComplexTransitionEnergyAndConjugateReverse)
{
    const std::vector<Complex> density = {1.0, Complex(0.25, 0.5), Complex(-0.5, 0.75), 0.5};
    const std::vector<Complex> h = {2.0, Complex(3.0, -1.0), Complex(3.0, 1.0), 4.0};
    const std::vector<Complex> reverse = {1.0, Complex(-0.5, -0.75), Complex(0.25, -0.5), 0.5};
    const Complex forward_energy = fde::one_body_energy_ry(density, h, 2);
    const Complex reverse_energy = fde::one_body_energy_ry(reverse, h, 2);
    const Complex expected(3.5, 4.5);
    const Complex expected_reverse(3.5, -4.5);
    expect_complex_near(forward_energy, expected);
    expect_complex_near(reverse_energy, expected_reverse);
}

TEST(FdeTransitionEnergyComponents, AppliesEnergyZeroShiftByTheMetricElectronCount)
{
    // cA=(1,0), cB=(0.8,0.6i), S01=0.2i give this P and Tr(SP)=1.
    const double real_density = 20.0 / 17.0;
    const double imaginary_density = 15.0 / 17.0;
    const Complex density_10(0.0, imaginary_density);
    const std::vector<Complex> density = {real_density, density_10, 0.0, 0.0};
    const std::vector<Complex> h = {2.0, 3.0, 3.0, 4.0};
    const std::vector<Complex> shifted_h = {10.0, Complex(3.0, -1.6), Complex(3.0, 1.6), 12.0};
    const Complex energy = fde::one_body_energy_ry(density, h, 2);
    const Complex shifted = fde::one_body_energy_ry(density, shifted_h, 2);
    const Complex shift = shifted - energy;
    expect_complex_near(shift, 8.0);
}

TEST(FdeTransitionEnergyComponents, RejectsMalformedNonfiniteAndOverflowingInputs)
{
    const std::vector<double> one = {1.0};
    const std::vector<double> empty;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const double infinity = std::numeric_limits<double>::infinity();
    const std::vector<double> infinite = {infinity};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> invalid = {nan};
    const double large = std::numeric_limits<double>::max();
    const std::vector<double> enormous = {large};
    const std::vector<Complex> complex_one = {1.0};
    const Complex invalid_imaginary(1.0, infinity);
    const std::vector<Complex> complex_invalid = {invalid_imaginary};
    EXPECT_THROW(fde::one_body_energy_ry(one, one, 0), std::invalid_argument);
    EXPECT_THROW(fde::one_body_energy_ry(one, one, maximum), std::invalid_argument);
    EXPECT_THROW(fde::one_body_energy_ry(empty, one, 1), std::invalid_argument);
    EXPECT_THROW(fde::one_body_energy_ry(one, infinite, 1), std::invalid_argument);
    EXPECT_THROW(fde::one_body_energy_ry(invalid, one, 1), std::invalid_argument);
    EXPECT_THROW(fde::one_body_energy_ry(complex_invalid, complex_one, 1), std::invalid_argument);
    EXPECT_THROW(fde::one_body_energy_ry(enormous, enormous, 1), std::overflow_error);
}

TEST(FdeTransitionEnergyComponents, HartreeKeepsTheComplexBilinearEnergyAndReverseConjugacy)
{
    // K=[[2,-2],[-2,2]], dV=1/4: E[a]=1, E[b]=9/4, integral a Kb=-3.
    const std::vector<double> real_density = {1.0, 3.0};
    const std::vector<double> imaginary_density = {2.0, -1.0};
    const std::vector<double> real_potential = {-4.0, 4.0};
    const std::vector<double> imaginary_potential = {6.0, -6.0};
    const std::vector<Complex> density = {Complex(1.0, 2.0), Complex(3.0, -1.0)};
    const std::vector<Complex> potential = {Complex(-4.0, 6.0), Complex(4.0, -6.0)};
    const std::vector<Complex> reverse_density = {Complex(1.0, -2.0), Complex(3.0, 1.0)};
    const std::vector<Complex> reverse_potential = {Complex(-4.0, -6.0), Complex(4.0, 6.0)};
    const double real_energy = fde::local_hartree_energy_ry(real_density, real_potential, 0.25);
    const double imaginary_energy = fde::local_hartree_energy_ry(imaginary_density, imaginary_potential, 0.25);
    const Complex energy = fde::local_hartree_energy_ry(density, potential, 0.25);
    const Complex reverse = fde::local_hartree_energy_ry(reverse_density, reverse_potential, 0.25);
    EXPECT_NEAR(real_energy, 1.0, 1.0e-12);
    EXPECT_NEAR(imaginary_energy, 2.25, 1.0e-12);
    const Complex expected(-1.25, -3.0);
    const Complex expected_reverse(-1.25, 3.0);
    expect_complex_near(energy, expected);
    expect_complex_near(reverse, expected_reverse);
}

TEST(FdeTransitionEnergyComponents, HartreeHasTheRealGammaLimitAndAnAdditiveLocalGridPartition)
{
    const std::vector<double> density = {1.0, 3.0};
    const std::vector<double> potential = {-4.0, 4.0};
    const std::vector<Complex> gamma_density = {1.0, 3.0};
    const std::vector<Complex> gamma_potential = {-4.0, 4.0};
    const std::vector<double> left_density = {1.0};
    const std::vector<double> left_potential = {-4.0};
    const std::vector<double> right_density = {3.0};
    const std::vector<double> right_potential = {4.0};
    const std::vector<double> empty;
    const double total = fde::local_hartree_energy_ry(density, potential, 0.25);
    const double left = fde::local_hartree_energy_ry(left_density, left_potential, 0.25);
    const double right = fde::local_hartree_energy_ry(right_density, right_potential, 0.25);
    const double empty_rank = fde::local_hartree_energy_ry(empty, empty, 0.25);
    const Complex gamma = fde::local_hartree_energy_ry(gamma_density, gamma_potential, 0.25);
    EXPECT_NEAR(total, 1.0, 1.0e-12);
    EXPECT_NEAR(left + right + empty_rank, total, 1.0e-12);
    expect_complex_near(gamma, total);
}

TEST(FdeTransitionEnergyComponents, HartreeRejectsMalformedNonfiniteAndOverflowingInputs)
{
    const std::vector<double> one = {1.0};
    const std::vector<double> empty;
    const double infinity = std::numeric_limits<double>::infinity();
    const std::vector<double> infinite = {infinity};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> invalid = {nan};
    const double large = std::numeric_limits<double>::max();
    const std::vector<double> enormous = {large};
    const std::vector<Complex> complex_one = {1.0};
    const Complex invalid_imaginary(1.0, infinity);
    const std::vector<Complex> complex_invalid = {invalid_imaginary};
    EXPECT_THROW(fde::local_hartree_energy_ry(empty, one, 1.0), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(one, one, 0.0), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(one, one, -1.0), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(one, one, infinity), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(one, one, nan), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(one, infinite, 1.0), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(invalid, one, 1.0), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(complex_invalid, complex_one, 1.0), std::invalid_argument);
    EXPECT_THROW(fde::local_hartree_energy_ry(enormous, enormous, 1.0), std::overflow_error);
}
