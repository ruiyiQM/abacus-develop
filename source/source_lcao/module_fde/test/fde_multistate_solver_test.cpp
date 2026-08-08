#include "../fde_multistate_solver.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{

double metric(const std::vector<double>& first,
              const std::vector<double>& second,
              const std::vector<double>& overlap,
              const std::size_t dimension)
{
    double result = 0.0;
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            result += first[row] * overlap[row + column * dimension] * second[column];
        }
    }
    return result;
}

fde::NonorthogonalSolverControls controls(const double cutoff)
{
    return {cutoff, 1.0e-12, 1.0e-10};
}

} // namespace

TEST(FdeMultistateSolver, SolvesGeneralizedProblemInCanonicalOrthogonalBasis)
{
    const fde::NonorthogonalStateProblem problem{{"A", "B"},
                                                 {1.0, 0.2, 0.2, 2.0},
                                                 {1.0, 0.1, 0.1, 1.0}};
    const fde::NonorthogonalStateSolution result
        = fde::NonorthogonalMultistateSolver::solve(problem, controls(1.0e-10));

    ASSERT_EQ(result.eigenvalues_ry.size(), 2);
    EXPECT_EQ(result.discarded_dimensions, 0);
    EXPECT_LT(result.maximum_residual, 1.0e-10);
    for (std::size_t first = 0; first < 2; ++first)
    {
        const std::vector<double> first_column
            = {result.coefficients[first * 2], result.coefficients[1 + first * 2]};
        for (std::size_t second = 0; second < 2; ++second)
        {
            const std::vector<double> second_column
                = {result.coefficients[second * 2], result.coefficients[1 + second * 2]};
            EXPECT_NEAR(metric(first_column, second_column, problem.overlap, 2),
                        first == second ? 1.0 : 0.0,
                        1.0e-12);
        }
    }
}

TEST(FdeMultistateSolver, DiscardsLinearlyDependentStateDirection)
{
    const fde::NonorthogonalStateProblem problem{{"A", "B"},
                                                 {1.5, 1.5, 1.5, 1.5},
                                                 {1.0, 1.0, 1.0, 1.0}};
    const fde::NonorthogonalStateSolution result
        = fde::NonorthogonalMultistateSolver::solve(problem, controls(1.0e-8));

    ASSERT_EQ(result.eigenvalues_ry.size(), 1);
    EXPECT_EQ(result.discarded_dimensions, 1);
    EXPECT_NEAR(result.eigenvalues_ry[0], 1.5, 1.0e-12);
}

TEST(FdeMultistateSolver, TracksCrossingRootsByGlobalMaximumOverlapAndAlignsPhase)
{
    fde::NonorthogonalStateSolution previous;
    previous.diabatic_dimension = 2;
    previous.eigenvalues_ry = {0.0, 1.0};
    previous.coefficients = {1.0, 0.0, 0.0, 1.0};

    fde::NonorthogonalStateSolution current;
    current.diabatic_dimension = 2;
    current.eigenvalues_ry = {0.9, 0.1};
    current.coefficients = {0.0, -1.0, 1.0, 0.0};

    const fde::RootTrackingResult tracked
        = fde::NonorthogonalMultistateSolver::track_roots(previous,
                                                         current,
                                                         {1.0, 0.0, 0.0, 1.0},
                                                         0.8);
    EXPECT_EQ(tracked.current_root_for_previous,
              (std::vector<std::size_t>{1, 0}));
    EXPECT_EQ(tracked.eigenvalues_ry, (std::vector<double>{0.1, 0.9}));
    EXPECT_EQ(tracked.phase_aligned_coefficients,
              (std::vector<double>{1.0, 0.0, 0.0, 1.0}));
    EXPECT_EQ(tracked.signed_root_overlaps,
              (std::vector<double>{1.0, -1.0}));
}
