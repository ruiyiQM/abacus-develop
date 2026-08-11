#include "source_lcao/module_fde/embedding/fde_subspace_solver.h"

#include "source_lcao/module_fde/embedding/fde_ao_projection.h"
#include "source_base/module_external/lapack_connector.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fde
{

namespace
{

void validate_symmetric_matrix(const std::vector<double>& matrix,
                               const std::size_t dimension,
                               const double tolerance,
                               const char* description)
{
    if (matrix.size() != dimension * dimension)
    {
        throw std::invalid_argument(std::string("FDE ") + description + " matrix size is inconsistent");
    }
    if (!std::isfinite(tolerance) || tolerance < 0.0)
    {
        throw std::invalid_argument("FDE matrix symmetry tolerance must be finite and nonnegative");
    }
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            const double value = matrix[row + column * dimension];
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(std::string("FDE ") + description
                                            + " matrix contains a non-finite value");
            }
            if (std::fabs(value - matrix[column + row * dimension]) > tolerance)
            {
                throw std::invalid_argument(std::string("FDE ") + description
                                            + " matrix must be symmetric");
            }
        }
    }
}

} // namespace

ActiveSubspaceProblem SubspaceSolver::build_problem(
    const std::vector<double>& full_hamiltonian_ry,
    const std::vector<double>& full_overlap,
    const std::size_t full_dimension,
    const std::vector<std::size_t>& active_orbitals,
    const bool contains_all_nuclear_operators,
    const double symmetry_tolerance)
{
    if (!contains_all_nuclear_operators)
    {
        throw std::invalid_argument(
            "FDE full Hamiltonian must include local and nonlocal operators from every nucleus");
    }
    validate_symmetric_matrix(full_hamiltonian_ry,
                              full_dimension,
                              symmetry_tolerance,
                              "full Hamiltonian");
    validate_symmetric_matrix(full_overlap, full_dimension, symmetry_tolerance, "full overlap");
    ActiveAoProjection::validate(full_dimension, active_orbitals);

    ActiveSubspaceProblem problem;
    problem.full_dimension = full_dimension;
    problem.active_orbitals = active_orbitals;
    problem.hamiltonian_ry = ActiveAoProjection::principal_submatrix(full_hamiltonian_ry,
                                                                     full_dimension,
                                                                     active_orbitals);
    problem.overlap = ActiveAoProjection::principal_submatrix(full_overlap,
                                                              full_dimension,
                                                              active_orbitals);
    return problem;
}

std::vector<double> SubspaceSolver::expand_active_matrix(
    const std::vector<double>& active_matrix,
    const std::size_t full_dimension,
    const std::vector<std::size_t>& active_orbitals)
{
    ActiveAoProjection::validate(full_dimension, active_orbitals);
    const std::size_t active_dimension = active_orbitals.size();
    if (active_matrix.size() != active_dimension * active_dimension)
    {
        throw std::invalid_argument("FDE active matrix size is inconsistent with its AO map");
    }

    std::vector<double> expanded(full_dimension * full_dimension, 0.0);
    for (std::size_t column = 0; column < active_dimension; ++column)
    {
        for (std::size_t row = 0; row < active_dimension; ++row)
        {
            expanded[active_orbitals[row] + active_orbitals[column] * full_dimension]
                = active_matrix[row + column * active_dimension];
        }
    }
    return expanded;
}

double SubspaceSolver::electron_count(const std::vector<double>& density_matrix,
                                      const std::vector<double>& overlap,
                                      const std::size_t dimension)
{
    if (density_matrix.size() != dimension * dimension || overlap.size() != dimension * dimension)
    {
        throw std::invalid_argument("FDE density and overlap matrix dimensions are inconsistent");
    }
    double count = 0.0;
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            count += density_matrix[row + column * dimension]
                     * overlap[column + row * dimension];
        }
    }
    return count;
}

ActiveSubspaceSolution SubspaceSolver::solve_serial(const ActiveSubspaceProblem& problem,
                                                    const std::vector<double>& occupations)
{
    const std::size_t active_dimension = problem.active_orbitals.size();
    ActiveAoProjection::validate(problem.full_dimension, problem.active_orbitals);
    validate_symmetric_matrix(problem.hamiltonian_ry, active_dimension, 1.0e-10, "active Hamiltonian");
    validate_symmetric_matrix(problem.overlap, active_dimension, 1.0e-10, "active overlap");
    if (occupations.size() != active_dimension)
    {
        throw std::invalid_argument("FDE serial subspace solver requires one occupation per eigenstate");
    }
    for (std::size_t state = 0; state < occupations.size(); ++state)
    {
        if (!std::isfinite(occupations[state]) || occupations[state] < 0.0
            || occupations[state] > 1.0)
        {
            throw std::invalid_argument("FDE spin-channel occupations must be between zero and one");
        }
    }

    ActiveSubspaceSolution solution;
    solution.eigenvalues_ry.assign(active_dimension, 0.0);
    solution.eigenvectors = problem.hamiltonian_ry;
    std::vector<double> overlap = problem.overlap;
    solution.occupations = occupations;

    const int itype = 1;
    const char jobz = 'V';
    const char uplo = 'U';
    const int dimension = static_cast<int>(active_dimension);
    const int leading_dimension = dimension;
    int workspace_size = -1;
    int info = 0;
    double workspace_query = 0.0;
    dsygv_(&itype,
           &jobz,
           &uplo,
           &dimension,
           solution.eigenvectors.data(),
           &leading_dimension,
           overlap.data(),
           &leading_dimension,
           solution.eigenvalues_ry.data(),
           &workspace_query,
           &workspace_size,
           &info);
    if (info != 0)
    {
        throw std::runtime_error("FDE generalized eigensolver workspace query failed");
    }
    workspace_size = std::max(1, static_cast<int>(workspace_query));
    std::vector<double> workspace(static_cast<std::size_t>(workspace_size), 0.0);
    dsygv_(&itype,
           &jobz,
           &uplo,
           &dimension,
           solution.eigenvectors.data(),
           &leading_dimension,
           overlap.data(),
           &leading_dimension,
           solution.eigenvalues_ry.data(),
           workspace.data(),
           &workspace_size,
           &info);
    if (info < 0)
    {
        throw std::runtime_error("FDE generalized eigensolver received an invalid argument");
    }
    if (info > dimension)
    {
        throw std::runtime_error("FDE active overlap matrix is not positive definite");
    }
    if (info != 0)
    {
        throw std::runtime_error("FDE generalized eigensolver did not converge");
    }

    solution.active_density_matrix.assign(active_dimension * active_dimension, 0.0);
    for (std::size_t state = 0; state < active_dimension; ++state)
    {
        for (std::size_t column = 0; column < active_dimension; ++column)
        {
            for (std::size_t row = 0; row < active_dimension; ++row)
            {
                solution.active_density_matrix[row + column * active_dimension]
                    += occupations[state]
                       * solution.eigenvectors[row + state * active_dimension]
                       * solution.eigenvectors[column + state * active_dimension];
            }
        }
    }
    solution.full_density_matrix = SubspaceSolver::expand_active_matrix(
        solution.active_density_matrix, problem.full_dimension, problem.active_orbitals);
    return solution;
}

} // namespace fde
