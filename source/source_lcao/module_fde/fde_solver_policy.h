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
                         int kpar);
};

} // namespace fde

#endif // FDE_SOLVER_POLICY_H
