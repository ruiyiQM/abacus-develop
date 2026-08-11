#include "../restart/fde_warm_start.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace
{

TEST(FdeWarmStart, UsesDensitySeedForFirstRequest)
{
    const fde::FdeWarmStartPlan plan = fde::FdeWarmStart::plan(0);
    EXPECT_EQ(plan.ionic_step, 0);
    EXPECT_FALSE(plan.reuse_ao_density_matrix);
    EXPECT_FALSE(plan.reuse_orbitals);
    EXPECT_STREQ(plan.mode, "density_seed");
}

TEST(FdeWarmStart, ReusesResidentElectronicStateAfterFirstRequest)
{
    const fde::FdeWarmStartPlan plan = fde::FdeWarmStart::plan(2);
    EXPECT_EQ(plan.ionic_step, 2);
    EXPECT_TRUE(plan.reuse_ao_density_matrix);
    EXPECT_TRUE(plan.reuse_orbitals);
    EXPECT_STREQ(plan.mode, "resident_ao_density_matrix_and_orbitals");
}

TEST(FdeWarmStart, RejectsIonicStepOverflow)
{
    const std::size_t too_large
        = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1U;
    EXPECT_THROW(fde::FdeWarmStart::plan(too_large), std::overflow_error);
}

} // namespace
