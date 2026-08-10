#ifndef FDE_SOLVER_POLICY_H
#define FDE_SOLVER_POLICY_H

#include <string>

namespace fde
{

class FdeSolverPolicy
{
  public:
    static bool is_distributed_solver(const std::string& solver);
    static void validate(const std::string& solver,
                         bool distributed_ao_matrices,
                         int kpar,
                         bool use_gpu);

    /**
     * Reset solver state that assumes the AO overlap storage persists between
     * diagonalizations. FDE constructs a fresh projected overlap for every
     * SCF iteration, so a cached factorization must never be reused.
     */
    static void prepare_fresh_overlap(const std::string& solver);
};

} // namespace fde

#endif // FDE_SOLVER_POLICY_H
