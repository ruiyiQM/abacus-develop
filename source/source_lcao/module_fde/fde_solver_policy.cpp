#include "fde_solver_policy.h"

#include <stdexcept>

namespace fde
{

bool FdeSolverPolicy::is_distributed_solver(const std::string& solver)
{
    return solver == "genelpa" || solver == "elpa" || solver == "scalapack_gvx";
}

void FdeSolverPolicy::validate(const std::string& solver,
                               const bool distributed_ao_matrices,
                               const int kpar)
{
    if (kpar != 1)
    {
        throw std::invalid_argument("FDE embedded_scf currently requires kpar 1");
    }
    if (solver == "lapack")
    {
        if (distributed_ao_matrices)
        {
            throw std::invalid_argument(
                "FDE distributed AO matrices require genelpa, elpa, or scalapack_gvx");
        }
        return;
    }
    if (!FdeSolverPolicy::is_distributed_solver(solver))
    {
        throw std::invalid_argument(
            "FDE embedded_scf supports lapack, genelpa, elpa, or scalapack_gvx");
    }
#ifndef __ELPA
    if (solver == "genelpa" || solver == "elpa")
    {
        throw std::invalid_argument(
            "FDE requested an ELPA solver but ABACUS was built without ELPA");
    }
#endif
}

} // namespace fde
