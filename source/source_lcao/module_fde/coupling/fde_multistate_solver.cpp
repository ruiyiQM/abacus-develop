#include "source_lcao/module_fde/coupling/fde_multistate_solver.h"

#include "source_base/module_external/lapack_connector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fde
{

namespace
{

void validate_symmetric(const std::vector<double>& matrix,
                        const std::size_t dimension,
                        const double tolerance,
                        const char* description)
{
    if (matrix.size() != dimension * dimension)
    {
        throw std::invalid_argument(std::string("FDE multistate ") + description
                                    + " dimensions are inconsistent");
    }
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            const double value = matrix[row + column * dimension];
            if (!std::isfinite(value)
                || std::fabs(value - matrix[column + row * dimension]) > tolerance)
            {
                throw std::invalid_argument(std::string("FDE multistate ") + description
                                            + " must be finite and symmetric");
            }
        }
    }
}

void diagonalize_symmetric(std::vector<double>& matrix,
                           std::vector<double>& eigenvalues,
                           const std::size_t dimension)
{
    eigenvalues.assign(dimension, 0.0);
    const char jobz = 'V';
    const char uplo = 'U';
    const int size = static_cast<int>(dimension);
    const int leading_dimension = size;
    int workspace_size = -1;
    int info = 0;
    double workspace_query = 0.0;
    dsyev_(&jobz,
           &uplo,
           &size,
           matrix.data(),
           &leading_dimension,
           eigenvalues.data(),
           &workspace_query,
           &workspace_size,
           &info);
    if (info != 0)
    {
        throw std::runtime_error("FDE multistate eigensolver workspace query failed");
    }
    workspace_size = std::max(1, static_cast<int>(workspace_query));
    std::vector<double> workspace(static_cast<std::size_t>(workspace_size), 0.0);
    dsyev_(&jobz,
           &uplo,
           &size,
           matrix.data(),
           &leading_dimension,
           eigenvalues.data(),
           workspace.data(),
           &workspace_size,
           &info);
    if (info < 0)
    {
        throw std::runtime_error("FDE multistate eigensolver received an invalid argument");
    }
    if (info > 0)
    {
        throw std::runtime_error("FDE multistate eigensolver did not converge");
    }
}

std::vector<std::size_t> maximum_weight_assignment(const std::vector<double>& weights,
                                                   const std::size_t dimension)
{
    std::vector<double> u(dimension + 1, 0.0);
    std::vector<double> v(dimension + 1, 0.0);
    std::vector<std::size_t> matched_row(dimension + 1, 0);
    std::vector<std::size_t> path(dimension + 1, 0);
    for (std::size_t row = 1; row <= dimension; ++row)
    {
        matched_row[0] = row;
        std::size_t column0 = 0;
        std::vector<double> minimum(dimension + 1,
                                    std::numeric_limits<double>::infinity());
        std::vector<bool> used(dimension + 1, false);
        do
        {
            used[column0] = true;
            const std::size_t row0 = matched_row[column0];
            double delta = std::numeric_limits<double>::infinity();
            std::size_t column1 = 0;
            for (std::size_t column = 1; column <= dimension; ++column)
            {
                if (used[column])
                {
                    continue;
                }
                const double cost = -weights[(row0 - 1) + (column - 1) * dimension];
                const double reduced = cost - u[row0] - v[column];
                if (reduced < minimum[column])
                {
                    minimum[column] = reduced;
                    path[column] = column0;
                }
                if (minimum[column] < delta)
                {
                    delta = minimum[column];
                    column1 = column;
                }
            }
            for (std::size_t column = 0; column <= dimension; ++column)
            {
                if (used[column])
                {
                    u[matched_row[column]] += delta;
                    v[column] -= delta;
                }
                else
                {
                    minimum[column] -= delta;
                }
            }
            column0 = column1;
        } while (matched_row[column0] != 0);

        do
        {
            const std::size_t column1 = path[column0];
            matched_row[column0] = matched_row[column1];
            column0 = column1;
        } while (column0 != 0);
    }

    std::vector<std::size_t> assignment(dimension, 0);
    for (std::size_t column = 1; column <= dimension; ++column)
    {
        assignment[matched_row[column] - 1] = column - 1;
    }
    return assignment;
}

double root_overlap(const NonorthogonalStateSolution& previous,
                    const NonorthogonalStateSolution& current,
                    const std::vector<double>& cross_overlap,
                    const std::size_t previous_root,
                    const std::size_t current_root)
{
    double result = 0.0;
    for (std::size_t current_state = 0;
         current_state < current.diabatic_dimension;
         ++current_state)
    {
        for (std::size_t previous_state = 0;
             previous_state < previous.diabatic_dimension;
             ++previous_state)
        {
            result += previous.coefficients[previous_state
                                            + previous_root * previous.diabatic_dimension]
                      * cross_overlap[previous_state
                                      + current_state * previous.diabatic_dimension]
                      * current.coefficients[current_state
                                             + current_root * current.diabatic_dimension];
        }
    }
    return result;
}

} // namespace

NonorthogonalStateSolution NonorthogonalMultistateSolver::solve(
    const NonorthogonalStateProblem& problem,
    const NonorthogonalSolverControls& controls)
{
    const std::size_t dimension = problem.state_labels.size();
    if (dimension == 0
        || !std::isfinite(controls.overlap_eigenvalue_cutoff)
        || controls.overlap_eigenvalue_cutoff <= 0.0
        || !std::isfinite(controls.symmetry_tolerance)
        || controls.symmetry_tolerance < 0.0
        || !std::isfinite(controls.residual_tolerance)
        || controls.residual_tolerance <= 0.0)
    {
        throw std::invalid_argument("FDE nonorthogonal multistate controls are invalid");
    }
    std::set<std::string> labels;
    for (std::size_t index = 0; index < dimension; ++index)
    {
        if (problem.state_labels[index].empty()
            || problem.state_labels[index].find_first_of(" \t\r\n") != std::string::npos
            || !labels.insert(problem.state_labels[index]).second)
        {
            throw std::invalid_argument("FDE multistate labels must be unique tokens");
        }
    }
    validate_symmetric(problem.hamiltonian_ry,
                       dimension,
                       controls.symmetry_tolerance,
                       "Hamiltonian");
    validate_symmetric(problem.overlap,
                       dimension,
                       controls.symmetry_tolerance,
                       "overlap");

    std::vector<double> overlap_vectors = problem.overlap;
    std::vector<double> overlap_eigenvalues;
    diagonalize_symmetric(overlap_vectors, overlap_eigenvalues, dimension);
    std::vector<std::size_t> retained;
    for (std::size_t index = 0; index < dimension; ++index)
    {
        if (overlap_eigenvalues[index] > controls.overlap_eigenvalue_cutoff)
        {
            retained.push_back(index);
        }
        else if (overlap_eigenvalues[index] < -controls.symmetry_tolerance)
        {
            throw std::invalid_argument("FDE multistate overlap has a negative eigenvalue");
        }
    }
    if (retained.empty())
    {
        throw std::runtime_error("FDE multistate overlap cutoff discarded every state");
    }

    const std::size_t rank = retained.size();
    std::vector<double> orthogonalizer(dimension * rank, 0.0);
    for (std::size_t column = 0; column < rank; ++column)
    {
        const std::size_t eigenvector = retained[column];
        const double scale = 1.0 / std::sqrt(overlap_eigenvalues[eigenvector]);
        for (std::size_t row = 0; row < dimension; ++row)
        {
            orthogonalizer[row + column * dimension]
                = overlap_vectors[row + eigenvector * dimension] * scale;
        }
    }

    std::vector<double> orthogonal_hamiltonian(rank * rank, 0.0);
    for (std::size_t column = 0; column < rank; ++column)
    {
        for (std::size_t row = 0; row < rank; ++row)
        {
            double value = 0.0;
            for (std::size_t right = 0; right < dimension; ++right)
            {
                for (std::size_t left = 0; left < dimension; ++left)
                {
                    value += orthogonalizer[left + row * dimension]
                             * problem.hamiltonian_ry[left + right * dimension]
                             * orthogonalizer[right + column * dimension];
                }
            }
            orthogonal_hamiltonian[row + column * rank] = value;
        }
    }

    NonorthogonalStateSolution solution;
    solution.diabatic_dimension = dimension;
    diagonalize_symmetric(orthogonal_hamiltonian,
                          solution.eigenvalues_ry,
                          rank);
    solution.coefficients.assign(dimension * rank, 0.0);
    for (std::size_t root = 0; root < rank; ++root)
    {
        for (std::size_t state = 0; state < dimension; ++state)
        {
            for (std::size_t canonical = 0; canonical < rank; ++canonical)
            {
                solution.coefficients[state + root * dimension]
                    += orthogonalizer[state + canonical * dimension]
                       * orthogonal_hamiltonian[canonical + root * rank];
            }
        }
    }
    solution.retained_overlap_eigenvalues.reserve(rank);
    for (std::size_t index = 0; index < rank; ++index)
    {
        solution.retained_overlap_eigenvalues.push_back(
            overlap_eigenvalues[retained[index]]);
    }
    solution.discarded_dimensions = dimension - rank;
    solution.maximum_residual = 0.0;
    for (std::size_t root = 0; root < rank; ++root)
    {
        std::vector<double> full_residual(dimension, 0.0);
        for (std::size_t row = 0; row < dimension; ++row)
        {
            double hc = 0.0;
            double sc = 0.0;
            for (std::size_t column = 0; column < dimension; ++column)
            {
                hc += problem.hamiltonian_ry[row + column * dimension]
                      * solution.coefficients[column + root * dimension];
                sc += problem.overlap[row + column * dimension]
                      * solution.coefficients[column + root * dimension];
            }
            full_residual[row] = hc - solution.eigenvalues_ry[root] * sc;
        }
        for (std::size_t canonical = 0; canonical < rank; ++canonical)
        {
            double projected_residual = 0.0;
            for (std::size_t state = 0; state < dimension; ++state)
            {
                projected_residual += orthogonalizer[state + canonical * dimension]
                                      * full_residual[state];
            }
            solution.maximum_residual
                = std::max(solution.maximum_residual, std::fabs(projected_residual));
        }
    }
    if (solution.maximum_residual > controls.residual_tolerance)
    {
        throw std::runtime_error("FDE multistate generalized eigenproblem residual is too large");
    }
    return solution;
}

RootTrackingResult NonorthogonalMultistateSolver::track_roots(
    const NonorthogonalStateSolution& previous,
    const NonorthogonalStateSolution& current,
    const std::vector<double>& previous_current_diabatic_overlap,
    const double minimum_root_overlap)
{
    const std::size_t root_count = previous.eigenvalues_ry.size();
    if (root_count == 0 || current.eigenvalues_ry.size() != root_count
        || previous.coefficients.size() != previous.diabatic_dimension * root_count
        || current.coefficients.size() != current.diabatic_dimension * root_count
        || previous_current_diabatic_overlap.size()
               != previous.diabatic_dimension * current.diabatic_dimension
        || !std::isfinite(minimum_root_overlap) || minimum_root_overlap < 0.0
        || minimum_root_overlap > 1.0)
    {
        throw std::invalid_argument("FDE root-tracking dimensions or controls are invalid");
    }

    std::vector<double> signed_overlaps(root_count * root_count, 0.0);
    std::vector<double> weights(root_count * root_count, 0.0);
    for (std::size_t current_root = 0; current_root < root_count; ++current_root)
    {
        for (std::size_t previous_root = 0; previous_root < root_count; ++previous_root)
        {
            const double value = root_overlap(previous,
                                              current,
                                              previous_current_diabatic_overlap,
                                              previous_root,
                                              current_root);
            signed_overlaps[previous_root + current_root * root_count] = value;
            weights[previous_root + current_root * root_count] = std::fabs(value);
        }
    }
    const std::vector<std::size_t> assignment
        = maximum_weight_assignment(weights, root_count);

    RootTrackingResult result;
    result.current_root_for_previous = assignment;
    result.signed_root_overlaps.resize(root_count);
    result.eigenvalues_ry.resize(root_count);
    result.phase_aligned_coefficients.assign(current.diabatic_dimension * root_count, 0.0);
    for (std::size_t previous_root = 0; previous_root < root_count; ++previous_root)
    {
        const std::size_t current_root = assignment[previous_root];
        const double overlap = signed_overlaps[previous_root + current_root * root_count];
        if (!std::isfinite(overlap) || std::fabs(overlap) < minimum_root_overlap)
        {
            throw std::runtime_error("FDE root tracking lost a state below its overlap threshold");
        }
        result.signed_root_overlaps[previous_root] = overlap;
        result.eigenvalues_ry[previous_root] = current.eigenvalues_ry[current_root];
        const double phase = overlap < 0.0 ? -1.0 : 1.0;
        for (std::size_t state = 0; state < current.diabatic_dimension; ++state)
        {
            result.phase_aligned_coefficients[state
                                              + previous_root * current.diabatic_dimension]
                = phase
                  * current.coefficients[state
                                         + current_root * current.diabatic_dimension];
        }
    }
    return result;
}

} // namespace fde
