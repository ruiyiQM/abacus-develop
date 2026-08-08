#include "../fde_ao_projection.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace
{

std::vector<double> diagonal_matrix(const std::vector<double>& diagonal)
{
    const std::size_t dimension = diagonal.size();
    std::vector<double> matrix(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index)
    {
        matrix[index + index * dimension] = diagonal[index];
    }
    return matrix;
}

} // namespace

TEST(FdeAoProjection, KeepsEnvironmentNonlocalOperatorInActiveSubspace)
{
    const std::size_t full_dimension = 4;
    const std::vector<std::size_t> active_orbitals{0, 1};
    std::vector<double> full_hamiltonian = diagonal_matrix({-1.0, -0.5, -0.4, -0.3});
    const std::vector<double> environment_nonlocal = diagonal_matrix({0.2, 0.1, 0.7, 0.8});
    for (std::size_t index = 0; index < full_hamiltonian.size(); ++index)
    {
        full_hamiltonian[index] += environment_nonlocal[index];
    }

    const std::vector<double> projected
        = fde::ActiveAoProjection::principal_submatrix(full_hamiltonian,
                                                       full_dimension,
                                                       active_orbitals);
    const std::vector<double> active_coefficients{1.0, 0.0};

    EXPECT_DOUBLE_EQ(projected[0], -0.8);
    EXPECT_DOUBLE_EQ(fde::ActiveAoProjection::quadratic_form(projected,
                                                            active_orbitals.size(),
                                                            active_coefficients),
                     -0.8);
}

TEST(FdeAoProjection, UsesColumnMajorPrincipalSubmatrix)
{
    const std::vector<double> full_matrix{
        0.0, 1.0, 2.0,
        3.0, 4.0, 5.0,
        6.0, 7.0, 8.0};
    const std::vector<double> projected
        = fde::ActiveAoProjection::principal_submatrix(full_matrix, 3, {0, 2});

    const std::vector<double> expected{0.0, 2.0, 6.0, 8.0};
    EXPECT_EQ(projected, expected);
}

TEST(FdeAoProjection, RejectsInvalidActiveOrbitalMaps)
{
    EXPECT_THROW(fde::ActiveAoProjection::validate(4, {}), std::invalid_argument);
    EXPECT_THROW(fde::ActiveAoProjection::validate(4, {0, 0}), std::invalid_argument);
    EXPECT_THROW(fde::ActiveAoProjection::validate(4, {2, 1}), std::invalid_argument);
    EXPECT_THROW(fde::ActiveAoProjection::validate(4, {0, 4}), std::out_of_range);
}
