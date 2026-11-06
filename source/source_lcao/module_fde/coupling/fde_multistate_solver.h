#ifndef FDE_COUPLING_MULTISTATE_SOLVER_H
#define FDE_COUPLING_MULTISTATE_SOLVER_H

#include <cstddef>
#include <complex>
#include <string>
#include <vector>

namespace fde
{

template <typename Scalar>
struct ScalarNonorthogonalStateProblem
{
    std::vector<std::string> state_labels;
    std::vector<Scalar> hamiltonian_ry;
    std::vector<Scalar> overlap;
};

using NonorthogonalStateProblem = ScalarNonorthogonalStateProblem<double>;
using ComplexNonorthogonalStateProblem = ScalarNonorthogonalStateProblem<std::complex<double>>;

struct NonorthogonalSolverControls
{
    double overlap_eigenvalue_cutoff;
    double symmetry_tolerance;
    double residual_tolerance;
};

template <typename Scalar>
struct ScalarNonorthogonalStateSolution
{
    std::size_t diabatic_dimension;
    std::vector<double> eigenvalues_ry;
    std::vector<Scalar> coefficients;
    std::vector<double> retained_overlap_eigenvalues;
    std::size_t discarded_dimensions;
    double maximum_residual;
};

using NonorthogonalStateSolution = ScalarNonorthogonalStateSolution<double>;
using ComplexNonorthogonalStateSolution = ScalarNonorthogonalStateSolution<std::complex<double>>;

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

#endif // FDE_COUPLING_MULTISTATE_SOLVER_H
