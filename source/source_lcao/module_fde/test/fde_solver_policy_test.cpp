#include "../fde_solver_policy.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(FdeSolverPolicy, KeepsLapackForReplicatedSerialMatrices)
{
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("lapack", false, 1));
    EXPECT_THROW(fde::FdeSolverPolicy::validate("lapack", true, 1),
                 std::invalid_argument);
}

TEST(FdeSolverPolicy, AcceptsDistributedScalapack)
{
    EXPECT_TRUE(fde::FdeSolverPolicy::is_distributed_solver("scalapack_gvx"));
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("scalapack_gvx", true, 1));
}

#ifdef __ELPA
TEST(FdeSolverPolicy, AcceptsDistributedElpaSolversWhenBuiltWithElpa)
{
    EXPECT_TRUE(fde::FdeSolverPolicy::is_distributed_solver("genelpa"));
    EXPECT_TRUE(fde::FdeSolverPolicy::is_distributed_solver("elpa"));
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("genelpa", true, 1));
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("elpa", true, 1));
}
#endif

TEST(FdeSolverPolicy, RejectsUnsupportedSolversAndKpointParallelism)
{
    EXPECT_THROW(fde::FdeSolverPolicy::validate("cg", false, 1),
                 std::invalid_argument);
    EXPECT_THROW(fde::FdeSolverPolicy::validate("scalapack_gvx", true, 2),
                 std::invalid_argument);
}
