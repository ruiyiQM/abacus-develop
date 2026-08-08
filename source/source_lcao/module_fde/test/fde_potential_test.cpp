#include "../fde_potential_evaluator.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

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
#else
    EXPECT_FALSE(fde::LibxcPbeProvider::available());
    EXPECT_THROW(provider.evaluate(active_density(),
                                   frozen_density(),
                                   {2, 1, 1, 0.5, 1.0, 1.0},
                                   1.0e-12),
                 std::runtime_error);
#endif
}
