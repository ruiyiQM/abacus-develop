#include "../fde_analytic_force.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{

fde::UniformGrid grid()
{
    return {2, 1, 1, 0.5, 1.0, 1.0};
}

fde::SpinDensity active_density()
{
    return {{0.8, 0.7}, {0.4, 0.3}};
}

fde::SpinDensity frozen_density()
{
    return {{0.2, 0.3}, {0.1, 0.2}};
}

fde::SpinDensity active_derivative()
{
    return {{0.03, -0.02}, {-0.01, 0.01}};
}

fde::SpinDensity frozen_derivative()
{
    return {{-0.02, 0.01}, {0.01, -0.02}};
}

fde::SpinDensity third_density()
{
    return {{0.15, 0.10}, {0.05, 0.08}};
}

fde::SpinDensity third_derivative()
{
    return {{0.01, -0.015}, {0.005, -0.005}};
}

fde::SpinDensity sum_density(const fde::SpinDensity& first,
                             const fde::SpinDensity& second)
{
    fde::SpinDensity result = first;
    for (std::size_t index = 0; index < result.alpha_bohr3.size(); ++index)
    {
        result.alpha_bohr3[index] += second.alpha_bohr3[index];
        result.beta_bohr3[index] += second.beta_bohr3[index];
    }
    return result;
}

std::vector<double> hartree_potential(const fde::SpinDensity& density)
{
    std::vector<double> result(density.alpha_bohr3.size(), 0.0);
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        result[index]
            = 0.4 * (density.alpha_bohr3[index] + density.beta_bohr3[index]);
    }
    return result;
}

fde::SpinDensity displaced(const fde::SpinDensity& density,
                           const fde::SpinDensity& derivative,
                           const double displacement)
{
    fde::SpinDensity result = density;
    for (std::size_t index = 0; index < result.alpha_bohr3.size(); ++index)
    {
        result.alpha_bohr3[index] += displacement * derivative.alpha_bohr3[index];
        result.beta_bohr3[index] += displacement * derivative.beta_bohr3[index];
    }
    return result;
}

double correction_energy(const fde::SpinDensity& active,
                         const fde::SpinDensity& frozen)
{
    const std::vector<double> frozen_hartree = hartree_potential(frozen);
    double hartree = 0.0;
    for (std::size_t index = 0; index < frozen_hartree.size(); ++index)
    {
        hartree += (active.alpha_bohr3[index] + active.beta_bohr3[index])
                   * frozen_hartree[index] * 0.5;
    }
    const double kinetic
        = fde::SemilocalFunctional::nonadditive_kinetic(
              active,
              frozen,
              grid(),
              fde::KineticFunctional::ThomasFermi,
              1.0e-12)
              .energy_ry;
    const double exchange
        = fde::SemilocalFunctional::nonadditive_dirac_exchange(
              active, frozen, grid(), 1.0e-12)
              .energy_ry;
    return hartree + kinetic + exchange;
}

double multifragment_correction_energy(
    const std::vector<fde::SpinDensity>& fragments)
{
    double energy = 0.0;
    fde::SpinDensity aggregate = fragments[0];
    for (std::size_t fragment = 1; fragment < fragments.size(); ++fragment)
    {
        const std::vector<double> fragment_hartree
            = hartree_potential(fragments[fragment]);
        for (std::size_t index = 0; index < fragment_hartree.size(); ++index)
        {
            energy += (aggregate.alpha_bohr3[index]
                       + aggregate.beta_bohr3[index])
                      * fragment_hartree[index] * 0.5;
        }
        energy += fde::SemilocalFunctional::nonadditive_kinetic(
                      aggregate,
                      fragments[fragment],
                      grid(),
                      fde::KineticFunctional::ThomasFermi,
                      1.0e-12)
                      .energy_ry;
        energy += fde::SemilocalFunctional::nonadditive_dirac_exchange(
                      aggregate, fragments[fragment], grid(), 1.0e-12)
                      .energy_ry;
        aggregate = sum_density(aggregate, fragments[fragment]);
    }
    return energy;
}

class DensityResponseBackend : public fde::LocalPotentialForceBackend
{
  public:
    std::vector<double> force_from_local_potential(
        const fde::SpinPotential& potential,
        const fde::SpinAoMatrix& density_matrix,
        const std::size_t full_ao_dimension,
        const std::size_t atom_count) const override
    {
        EXPECT_EQ(full_ao_dimension, 1);
        EXPECT_EQ(atom_count, 1);
        fde::SpinDensity derivative;
        if (density_matrix.alpha[0] < 1.5)
        {
            derivative = active_derivative();
        }
        else if (density_matrix.alpha[0] < 2.5)
        {
            derivative = frozen_derivative();
        }
        else if (density_matrix.alpha[0] < 3.5)
        {
            derivative = sum_density(active_derivative(), frozen_derivative());
        }
        else
        {
            derivative = third_derivative();
        }
        std::vector<double> force(3, 0.0);
        for (std::size_t index = 0; index < potential.alpha_ry.size(); ++index)
        {
            force[0]
                -= (potential.alpha_ry[index] * derivative.alpha_bohr3[index]
                    + potential.beta_ry[index] * derivative.beta_bohr3[index])
                   * 0.5;
        }
        return force;
    }
};

fde::DiagonalFdeForceRequest request()
{
    fde::DiagonalFdeForceRequest result;
    result.fragments
        = {{"A", active_density(), hartree_potential(active_density()), {{1.0}, {0.0}}},
           {"B", frozen_density(), hartree_potential(frozen_density()), {{2.0}, {0.0}}}};
    result.full_ao_dimension = 1;
    result.atom_count = 1;
    result.potential_config
        = {grid(), fde::KineticFunctional::ThomasFermi, 1.0e-12};
    result.base_force_ry_per_bohr = {0.3, -0.2, 0.1};
    result.hartree_reciprocity_tolerance_ry = 1.0e-12;
    return result;
}

} // namespace

TEST(FdeAnalyticForce, MatchesCentralDifferenceOfSemilocalCorrection)
{
    const fde::DiracExchangeProvider exchange;
    const DensityResponseBackend backend;
    const fde::DiagonalFdeForceResult result
        = fde::DiagonalFdeAnalyticForce::evaluate(request(), exchange, backend);

    const double step = 1.0e-6;
    const double energy_plus
        = correction_energy(displaced(active_density(), active_derivative(), step),
                            displaced(frozen_density(), frozen_derivative(), step));
    const double energy_minus
        = correction_energy(displaced(active_density(), active_derivative(), -step),
                            displaced(frozen_density(), frozen_derivative(), -step));
    const double finite_difference_force = -(energy_plus - energy_minus) / (2.0 * step);

    EXPECT_NEAR(result.total_force_ry_per_bohr[0]
                    - result.base_force_ry_per_bohr[0],
                finite_difference_force,
                2.0e-9);
    EXPECT_DOUBLE_EQ(result.total_force_ry_per_bohr[1], -0.2);
    EXPECT_DOUBLE_EQ(result.total_force_ry_per_bohr[2], 0.1);
    EXPECT_NEAR(result.total_force_ry_per_bohr[0],
                result.base_force_ry_per_bohr[0]
                    + result.hartree_cross_force_ry_per_bohr[0]
                    + result.nonadditive_kinetic_force_ry_per_bohr[0]
                    + result.nonadditive_xc_force_ry_per_bohr[0],
                1.0e-15);
}

TEST(FdeAnalyticForce, RejectsNonreciprocalHartreeInputs)
{
    fde::DiagonalFdeForceRequest invalid = request();
    invalid.fragments[0].hartree_potential_ry[0] += 0.1;
    const fde::DiracExchangeProvider exchange;
    const DensityResponseBackend backend;

    EXPECT_THROW(fde::DiagonalFdeAnalyticForce::evaluate(invalid, exchange, backend),
                 std::invalid_argument);
}

TEST(FdeAnalyticForce, SupportsMoreThanTwoFragments)
{
    fde::DiagonalFdeForceRequest multi = request();
    multi.fragments.push_back(
        {"C", third_density(), hartree_potential(third_density()), {{4.0}, {0.0}}});
    const fde::DiracExchangeProvider exchange;
    const DensityResponseBackend backend;
    const fde::DiagonalFdeForceResult result
        = fde::DiagonalFdeAnalyticForce::evaluate(multi, exchange, backend);

    const double step = 1.0e-6;
    const double energy_plus = multifragment_correction_energy(
        {displaced(active_density(), active_derivative(), step),
         displaced(frozen_density(), frozen_derivative(), step),
         displaced(third_density(), third_derivative(), step)});
    const double energy_minus = multifragment_correction_energy(
        {displaced(active_density(), active_derivative(), -step),
         displaced(frozen_density(), frozen_derivative(), -step),
         displaced(third_density(), third_derivative(), -step)});
    const double finite_difference_force = -(energy_plus - energy_minus) / (2.0 * step);

    EXPECT_NEAR(result.total_force_ry_per_bohr[0]
                    - result.base_force_ry_per_bohr[0],
                finite_difference_force,
                3.0e-9);
}
