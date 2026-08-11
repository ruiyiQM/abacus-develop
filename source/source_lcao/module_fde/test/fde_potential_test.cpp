#include "../embedding/fde_potential_evaluator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

fde::SpinDensity active_density()
{
    return {{0.8, 0.7}, {0.4, 0.3}};
}

fde::SpinDensity frozen_density()
{
    return {{0.2, 0.3}, {0.1, 0.2}};
}

fde::SpinDensity extended_density(const std::size_t size, const double offset)
{
    fde::SpinDensity density;
    density.alpha_bohr3.resize(size);
    density.beta_bohr3.resize(size);
    for (std::size_t index = 0; index < size; ++index)
    {
        density.alpha_bohr3[index] = offset + 0.01 * static_cast<double>(index % 17);
        density.beta_bohr3[index] = offset + 0.02 * static_cast<double>(index % 11);
    }
    return density;
}

#ifdef _OPENMP
class OpenmpThreadCountGuard
{
  public:
    OpenmpThreadCountGuard() : original_(omp_get_max_threads()) {}
    ~OpenmpThreadCountGuard() { omp_set_num_threads(original_); }

  private:
    int original_;
};
#endif

} // namespace

TEST(FdePotential, CombinesHartreeKineticAndXcWithoutNuclearPotential)
{
    const fde::PotFdeConfig config{{2, 1, 1, 0.5, 1.0, 1.0},
                                   fde::KineticFunctional::ThomasFermi,
                                   1.0e-12};
    const fde::DiracExchangeProvider xc;

    const fde::EmbeddingPotentialResult result
        = fde::EmbeddingPotentialEvaluator::evaluate(active_density(),
                                                     frozen_density(),
                                                     {0.4, 0.6},
                                                     config,
                                                     xc);

    EXPECT_NEAR(result.hartree_cross_energy_ry,
                ((0.8 + 0.4) * 0.4 + (0.7 + 0.3) * 0.6) * 0.5,
                1.0e-14);
    EXPECT_NE(result.nonadditive_kinetic_energy_ry, 0.0);
    EXPECT_NE(result.nonadditive_xc_energy_ry, 0.0);
    EXPECT_EQ(result.potential.alpha_ry.size(), 2);
    EXPECT_NE(result.potential.alpha_ry[0], 0.4);
}

TEST(FdePotential, PreparedFrozenCacheMatchesReferenceEvaluation)
{
    const fde::PotFdeConfig config{{2, 1, 1, 0.5, 1.0, 1.0},
                                   fde::KineticFunctional::ThomasFermi,
                                   1.0e-12};
    const fde::DiracExchangeProvider xc;
    const fde::EmbeddingPotentialResult reference
        = fde::EmbeddingPotentialEvaluator::evaluate(active_density(),
                                                     frozen_density(),
                                                     {0.4, 0.6},
                                                     config,
                                                     xc);
    const fde::FrozenEmbeddingCache cache
        = fde::EmbeddingPotentialEvaluator::prepare_frozen(
            frozen_density(), config, xc, nullptr);
    const fde::EmbeddingPotentialResult cached
        = fde::EmbeddingPotentialEvaluator::evaluate_cached(
            active_density(), frozen_density(), {0.4, 0.6}, config, xc, cache, nullptr);

    EXPECT_DOUBLE_EQ(cached.hartree_cross_energy_ry,
                     reference.hartree_cross_energy_ry);
    EXPECT_DOUBLE_EQ(cached.nonadditive_kinetic_energy_ry,
                     reference.nonadditive_kinetic_energy_ry);
    EXPECT_DOUBLE_EQ(cached.nonadditive_xc_energy_ry,
                     reference.nonadditive_xc_energy_ry);
    EXPECT_EQ(cached.potential.alpha_ry, reference.potential.alpha_ry);
    EXPECT_EQ(cached.potential.beta_ry, reference.potential.beta_ry);
}

TEST(FdePotential, LibxcPbeReportsBuildAvailability)
{
    const fde::LibxcPbeProvider provider;
#ifdef __LIBXC
    EXPECT_TRUE(fde::LibxcPbeProvider::available());
    const fde::NonadditiveFunctionalResult result
        = provider.evaluate(active_density(),
                            frozen_density(),
                            {2, 1, 1, 0.5, 1.0, 1.0},
                            1.0e-12);
    EXPECT_TRUE(std::isfinite(result.energy_ry));
    EXPECT_EQ(result.active_potential.alpha_ry.size(), 2);
    const fde::FrozenSemilocalCache cache
        = provider.prepare_frozen(frozen_density(),
                                  {2, 1, 1, 0.5, 1.0, 1.0},
                                  1.0e-12,
                                  nullptr);
    const fde::NonadditiveFunctionalResult cached
        = provider.evaluate_cached(active_density(),
                                   frozen_density(),
                                   {2, 1, 1, 0.5, 1.0, 1.0},
                                   1.0e-12,
                                   cache,
                                   nullptr);
    EXPECT_DOUBLE_EQ(cached.energy_ry, result.energy_ry);
    EXPECT_EQ(cached.active_potential.alpha_ry,
              result.active_potential.alpha_ry);
#else
    EXPECT_FALSE(fde::LibxcPbeProvider::available());
    EXPECT_THROW(provider.evaluate(active_density(),
                                   frozen_density(),
                                   {2, 1, 1, 0.5, 1.0, 1.0},
                                   1.0e-12),
                 std::runtime_error);
#endif
}

#if defined(__LIBXC) && defined(_OPENMP)
TEST(FdePotential, LibxcPbeIsThreadCountInvariant)
{
    OpenmpThreadCountGuard guard;
    const fde::UniformGrid grid{40, 40, 41, 0.35, 0.36, 0.37};
    const std::size_t size = grid.x * grid.y * grid.z;
    const fde::SpinDensity active = extended_density(size, 0.2);
    const fde::SpinDensity frozen = extended_density(size, 0.1);
    const fde::LibxcPbeProvider provider;

    omp_set_num_threads(1);
    const fde::NonadditiveFunctionalResult serial
        = provider.evaluate(active, frozen, grid, 1.0e-12);
    omp_set_num_threads(4);
    const fde::NonadditiveFunctionalResult parallel
        = provider.evaluate(active, frozen, grid, 1.0e-12);

    EXPECT_NEAR(serial.energy_ry, parallel.energy_ry,
                1.0e-11 * std::max(1.0, std::fabs(serial.energy_ry)));
    ASSERT_EQ(serial.active_potential.alpha_ry.size(),
              parallel.active_potential.alpha_ry.size());
    double maximum_alpha_difference = 0.0;
    double maximum_beta_difference = 0.0;
    for (std::size_t index = 0; index < size; ++index)
    {
        maximum_alpha_difference
            = std::max(maximum_alpha_difference,
                       std::fabs(serial.active_potential.alpha_ry[index]
                                 - parallel.active_potential.alpha_ry[index]));
        maximum_beta_difference
            = std::max(maximum_beta_difference,
                       std::fabs(serial.active_potential.beta_ry[index]
                                 - parallel.active_potential.beta_ry[index]));
    }
    EXPECT_LE(maximum_alpha_difference, 1.0e-12);
    EXPECT_LE(maximum_beta_difference, 1.0e-12);
}
#endif
