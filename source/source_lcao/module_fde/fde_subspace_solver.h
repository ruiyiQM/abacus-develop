#ifndef FDE_SUBSPACE_SOLVER_H
#define FDE_SUBSPACE_SOLVER_H

#include <cstddef>
#include <vector>

namespace fde
{

struct ActiveSubspaceProblem
{
    std::size_t full_dimension;
    std::vector<std::size_t> active_orbitals;
    std::vector<double> hamiltonian_ry;
    std::vector<double> overlap;
};

struct ActiveSubspaceSolution
{
    std::vector<double> eigenvalues_ry;
    std::vector<double> eigenvectors;
    std::vector<double> occupations;
    std::vector<double> active_density_matrix;
    std::vector<double> full_density_matrix;
};

class SubspaceSolver
{
  public:
    static ActiveSubspaceProblem build_problem(
        const std::vector<double>& full_hamiltonian_ry,
        const std::vector<double>& full_overlap,
        const std::size_t full_dimension,
        const std::vector<std::size_t>& active_orbitals,
        const bool contains_all_nuclear_operators,
        const double symmetry_tolerance);

    static ActiveSubspaceSolution solve_serial(const ActiveSubspaceProblem& problem,
                                               const std::vector<double>& occupations);

    static std::vector<double> expand_active_matrix(
        const std::vector<double>& active_matrix,
        const std::size_t full_dimension,
        const std::vector<std::size_t>& active_orbitals);

    static double electron_count(const std::vector<double>& density_matrix,
                                 const std::vector<double>& overlap,
                                 const std::size_t dimension);
};

} // namespace fde

#endif // FDE_SUBSPACE_SOLVER_H
