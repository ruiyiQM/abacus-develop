#include "../fde_semilocal_functional.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace
{

fde::UniformGrid line_grid()
{
    return {4, 1, 1, 0.5, 1.0, 1.0};
}

fde::SpinDensity zero_density()
{
    return {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
}

fde::SpinDensity active_density()
{
    return {{0.8, 0.7, 0.6, 0.7}, {0.4, 0.3, 0.2, 0.3}};
}

fde::SpinDensity frozen_density()
{
    return {{0.2, 0.3, 0.4, 0.3}, {0.1, 0.2, 0.3, 0.2}};
}

} // namespace

TEST(FdeSemilocalFunctional, ZeroFrozenDensityHasZeroNonadditiveTerms)
{
    const fde::NonadditiveFunctionalResult kinetic
        = fde::SemilocalFunctional::nonadditive_kinetic(active_density(),
                                                        zero_density(),
                                                        line_grid(),
                                                        fde::KineticFunctional::Lc94Pw91k,
                                                        1.0e-12);
    const fde::NonadditiveFunctionalResult exchange
        = fde::SemilocalFunctional::nonadditive_dirac_exchange(active_density(),
                                                               zero_density(),
                                                               line_grid(),
                                                               1.0e-12);

    EXPECT_NEAR(kinetic.energy_ry, 0.0, 1.0e-14);
    EXPECT_NEAR(exchange.energy_ry, 0.0, 1.0e-14);
    for (std::size_t index = 0; index < 4; ++index)
    {
        EXPECT_NEAR(kinetic.active_potential.alpha_ry[index], 0.0, 1.0e-13);
        EXPECT_NEAR(exchange.active_potential.beta_ry[index], 0.0, 1.0e-13);
    }
}

TEST(FdeSemilocalFunctional, Lc94ReducesToThomasFermiForUniformDensity)
{
    const fde::SpinDensity active{{0.5, 0.5, 0.5, 0.5}, {0.25, 0.25, 0.25, 0.25}};
    const fde::SpinDensity frozen{{0.2, 0.2, 0.2, 0.2}, {0.1, 0.1, 0.1, 0.1}};
    const fde::NonadditiveFunctionalResult tf
        = fde::SemilocalFunctional::nonadditive_kinetic(active,
                                                        frozen,
                                                        line_grid(),
                                                        fde::KineticFunctional::ThomasFermi,
                                                        1.0e-12);
    const fde::NonadditiveFunctionalResult lc94
        = fde::SemilocalFunctional::nonadditive_kinetic(active,
                                                        frozen,
                                                        line_grid(),
                                                        fde::KineticFunctional::Lc94Pw91k,
                                                        1.0e-12);

    EXPECT_NEAR(lc94.energy_ry, tf.energy_ry, 1.0e-13);
    EXPECT_NEAR(lc94.active_potential.alpha_ry[0],
                tf.active_potential.alpha_ry[0],
                1.0e-13);
}

TEST(FdeSemilocalFunctional, Lc94PotentialIsTheEnergyDerivative)
{
    fde::SpinDensity active = active_density();
    const fde::SpinDensity frozen = frozen_density();
    const double epsilon = 1.0e-6;
    const std::size_t varied_index = 1;
    const fde::NonadditiveFunctionalResult reference
        = fde::SemilocalFunctional::nonadditive_kinetic(active,
                                                        frozen,
                                                        line_grid(),
                                                        fde::KineticFunctional::Lc94Pw91k,
                                                        1.0e-12);
    active.alpha_bohr3[varied_index] += epsilon;
    const double energy_plus
        = fde::SemilocalFunctional::nonadditive_kinetic(active,
                                                        frozen,
                                                        line_grid(),
                                                        fde::KineticFunctional::Lc94Pw91k,
                                                        1.0e-12)
              .energy_ry;
    active.alpha_bohr3[varied_index] -= 2.0 * epsilon;
    const double energy_minus
        = fde::SemilocalFunctional::nonadditive_kinetic(active,
                                                        frozen,
                                                        line_grid(),
                                                        fde::KineticFunctional::Lc94Pw91k,
                                                        1.0e-12)
              .energy_ry;
    const double volume_element = 0.5;
    const double finite_difference = (energy_plus - energy_minus) / (2.0 * epsilon * volume_element);

    EXPECT_NEAR(reference.active_potential.alpha_ry[varied_index],
                finite_difference,
                2.0e-6);
}

TEST(FdeSemilocalFunctional, ExchangeIsSymmetricAndRejectsNegativeDensity)
{
    const fde::NonadditiveFunctionalResult first
        = fde::SemilocalFunctional::nonadditive_dirac_exchange(active_density(),
                                                               frozen_density(),
                                                               line_grid(),
                                                               1.0e-12);
    const fde::NonadditiveFunctionalResult second
        = fde::SemilocalFunctional::nonadditive_dirac_exchange(frozen_density(),
                                                               active_density(),
                                                               line_grid(),
                                                               1.0e-12);
    EXPECT_NEAR(first.energy_ry, second.energy_ry, 1.0e-14);

    fde::SpinDensity invalid = active_density();
    invalid.alpha_bohr3[0] = -1.0;
    EXPECT_THROW(fde::SemilocalFunctional::nonadditive_dirac_exchange(
                     invalid, frozen_density(), line_grid(), 1.0e-12),
                 std::invalid_argument);
}
