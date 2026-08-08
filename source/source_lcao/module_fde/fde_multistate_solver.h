#ifndef FDE_MULTISTATE_SOLVER_H
#define FDE_MULTISTATE_SOLVER_H

#include <cstddef>
#include <string>
#include <vector>

namespace fde
{

struct NonorthogonalStateProblem
{
    std::vector<std::string> state_labels;
    std::vector<double> hamiltonian_ry;
    std::vector<double> overlap;
};

struct NonorthogonalSolverControls
{
    double overlap_eigenvalue_cutoff;
    double symmetry_tolerance;
    double residual_tolerance;
};

struct NonorthogonalStateSolution
{
    std::size_t diabatic_dimension;
    std::vector<double> eigenvalues_ry;
    std::vector<double> coefficients;
    std::vector<double> retained_overlap_eigenvalues;
    std::size_t discarded_dimensions;
    double maximum_residual;
};

struct RootTrackingResult
{
    std::vector<std::size_t> current_root_for_previous;
    std::vector<double> signed_root_overlaps;
    std::vector<double> eigenvalues_ry;
    std::vector<double> phase_aligned_coefficients;
};

class NonorthogonalMultistateSolver
{
  public:
    static NonorthogonalStateSolution solve(const NonorthogonalStateProblem& problem,
                                            const NonorthogonalSolverControls& controls);

    static RootTrackingResult track_roots(
        const NonorthogonalStateSolution& previous,
        const NonorthogonalStateSolution& current,
        const std::vector<double>& previous_current_diabatic_overlap,
        double minimum_root_overlap);
};

} // namespace fde

#endif // FDE_MULTISTATE_SOLVER_H
