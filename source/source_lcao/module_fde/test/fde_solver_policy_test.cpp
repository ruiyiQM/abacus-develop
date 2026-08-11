#include "../runtime/fde_solver_policy.h"

#ifdef __ELPA
#include "source_hsolver/diago_elpa.h"
#include "source_hsolver/diago_elpa_native.h"
#endif

#include <gtest/gtest.h>

#include <stdexcept>

TEST(FdeSolverPolicy, KeepsLapackForReplicatedSerialMatrices)
{
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("lapack", false, 1, false));
    EXPECT_THROW(fde::FdeSolverPolicy::validate("lapack", true, 1, false),
                 std::invalid_argument);
}

TEST(FdeSolverPolicy, AcceptsDistributedScalapack)
{
    EXPECT_TRUE(fde::FdeSolverPolicy::is_distributed_solver("scalapack_gvx"));
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("scalapack_gvx", true, 1, false));
}

TEST(FdeSolverPolicy, RejectsCpuOnlySolverForGpuExecution)
{
    EXPECT_THROW(fde::FdeSolverPolicy::validate("genelpa", true, 1, true),
                 std::invalid_argument);
    EXPECT_THROW(fde::FdeSolverPolicy::validate("cusolver", true, 1, false),
                 std::invalid_argument);
}

#ifdef __CUDA
TEST(FdeSolverPolicy, AcceptsCusolverForDistributedGpuMatrices)
{
    EXPECT_TRUE(fde::FdeSolverPolicy::is_distributed_solver("cusolver"));
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("cusolver", true, 1, true));
}
#else
TEST(FdeSolverPolicy, RejectsCusolverWhenBuiltWithoutCuda)
{
    EXPECT_THROW(fde::FdeSolverPolicy::validate("cusolver", true, 1, true),
                 std::invalid_argument);
}
#endif

#ifdef __ELPA
TEST(FdeSolverPolicy, AcceptsDistributedElpaSolversWhenBuiltWithElpa)
{
    EXPECT_TRUE(fde::FdeSolverPolicy::is_distributed_solver("genelpa"));
    EXPECT_TRUE(fde::FdeSolverPolicy::is_distributed_solver("elpa"));
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("genelpa", true, 1, false));
    EXPECT_NO_THROW(fde::FdeSolverPolicy::validate("elpa", true, 1, false));
}

TEST(FdeSolverPolicy, ResetsElpaFactorizationForFreshProjectedOverlap)
{
    hsolver::DiagoElpa<double>::DecomposedState = 2;
    hsolver::DiagoElpaNative<double>::DecomposedState = 1;

    fde::FdeSolverPolicy::prepare_fresh_overlap("genelpa");
    EXPECT_EQ(hsolver::DiagoElpa<double>::DecomposedState, 0);
    EXPECT_EQ(hsolver::DiagoElpaNative<double>::DecomposedState, 1);

    fde::FdeSolverPolicy::prepare_fresh_overlap("elpa");
    EXPECT_EQ(hsolver::DiagoElpaNative<double>::DecomposedState, 0);
}
#endif

TEST(FdeSolverPolicy, RejectsUnsupportedSolversAndKpointParallelism)
{
    EXPECT_THROW(fde::FdeSolverPolicy::validate("cg", false, 1, false),
                 std::invalid_argument);
    EXPECT_THROW(fde::FdeSolverPolicy::validate("scalapack_gvx", true, 2, false),
                 std::invalid_argument);
}
