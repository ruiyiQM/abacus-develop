#include "../embedding/fde_subspace_solver.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace
{

std::vector<double> diagonal(const std::vector<double>& values)
{
    std::vector<double> matrix(values.size() * values.size(), 0.0);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        matrix[index + index * values.size()] = values[index];
    }
    return matrix;
}

} // namespace

TEST(FdeSubspaceSolver, SolvesActiveProblemAndExpandsDensity)
{
    const std::vector<double> full_hamiltonian = diagonal({-1.0, 2.0, -0.5});
    const std::vector<double> full_overlap = diagonal({1.0, 1.0, 1.0});
    const fde::ActiveSubspaceProblem problem
        = fde::SubspaceSolver::build_problem(full_hamiltonian,
                                             full_overlap,
                                             3,
                                             {0, 2},
                                             true,
                                             1.0e-12);
    const fde::ActiveSubspaceSolution solution
        = fde::SubspaceSolver::solve_serial(problem, {1.0, 0.0});

    ASSERT_EQ(solution.eigenvalues_ry.size(), 2);
    EXPECT_NEAR(solution.eigenvalues_ry[0], -1.0, 1.0e-12);
    EXPECT_NEAR(solution.eigenvalues_ry[1], -0.5, 1.0e-12);
    EXPECT_NEAR(fde::SubspaceSolver::electron_count(solution.active_density_matrix,
                                                    problem.overlap,
                                                    2),
                1.0,
                1.0e-12);
    EXPECT_DOUBLE_EQ(solution.full_density_matrix[1 + 1 * 3], 0.0);
    EXPECT_DOUBLE_EQ(solution.full_density_matrix[0 + 1 * 3], 0.0);
    EXPECT_DOUBLE_EQ(solution.full_density_matrix[1 + 2 * 3], 0.0);
}

TEST(FdeSubspaceSolver, PreservesNonorthogonalElectronCount)
{
    const std::vector<double> hamiltonian{-1.0, 0.1, 0.1, -0.4};
    const std::vector<double> overlap{1.0, 0.2, 0.2, 1.0};
    const fde::ActiveSubspaceProblem problem
        = fde::SubspaceSolver::build_problem(hamiltonian,
                                             overlap,
                                             2,
                                             {0, 1},
                                             true,
                                             1.0e-12);
    const fde::ActiveSubspaceSolution solution
        = fde::SubspaceSolver::solve_serial(problem, {1.0, 0.0});

    EXPECT_NEAR(fde::SubspaceSolver::electron_count(solution.active_density_matrix,
                                                    problem.overlap,
                                                    2),
                1.0,
                1.0e-12);
}

TEST(FdeSubspaceSolver, RequiresFullExternalOperatorAndPositiveOverlap)
{
    EXPECT_THROW(fde::SubspaceSolver::build_problem(diagonal({-1.0, -0.5}),
                                                    diagonal({1.0, 1.0}),
                                                    2,
                                                    {0},
                                                    false,
                                                    1.0e-12),
                 std::invalid_argument);

    const fde::ActiveSubspaceProblem invalid_overlap
        = fde::SubspaceSolver::build_problem(diagonal({-1.0, -0.5}),
                                             diagonal({1.0, -1.0}),
                                             2,
                                             {0, 1},
                                             true,
                                             1.0e-12);
    EXPECT_THROW(fde::SubspaceSolver::solve_serial(invalid_overlap, {1.0, 0.0}),
                 std::runtime_error);
}
