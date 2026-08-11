#include "source_lcao/module_fde/runtime/fde_solver_policy.h"

#ifdef __ELPA
#include "source_hsolver/diago_elpa.h"
#include "source_hsolver/diago_elpa_native.h"
#endif

#include <stdexcept>

namespace fde
{

bool FdeSolverPolicy::is_distributed_solver(const std::string& solver)
{
    return solver == "genelpa" || solver == "elpa" || solver == "scalapack_gvx"
           || solver == "cusolver";
}

void FdeSolverPolicy::validate(const std::string& solver,
                               const bool distributed_ao_matrices,
                               const int kpar,
                               const bool use_gpu)
{
    if (kpar != 1)
    {
        throw std::invalid_argument("FDE embedded_scf currently requires kpar 1");
    }
    if (use_gpu && solver != "cusolver" && solver != "elpa")
    {
        throw std::invalid_argument(
            "FDE device gpu requires ks_solver cusolver or GPU-enabled elpa");
    }
    if (solver == "cusolver")
    {
        if (!use_gpu)
        {
            throw std::invalid_argument(
                "FDE ks_solver cusolver requires device gpu");
        }
#ifndef __CUDA
        throw std::invalid_argument(
            "FDE requested cuSOLVER but ABACUS was built without CUDA");
#else
        return;
#endif
    }
    if (solver == "lapack")
    {
        if (distributed_ao_matrices)
        {
            throw std::invalid_argument(
                "FDE distributed AO matrices require genelpa, elpa, scalapack_gvx, or cusolver");
        }
        return;
    }
    if (!FdeSolverPolicy::is_distributed_solver(solver))
    {
        throw std::invalid_argument(
            "FDE embedded_scf supports lapack, genelpa, elpa, scalapack_gvx, or cusolver");
    }
#ifndef __ELPA
    if (solver == "genelpa" || solver == "elpa")
    {
        throw std::invalid_argument(
            "FDE requested an ELPA solver but ABACUS was built without ELPA");
    }
#endif
}

void FdeSolverPolicy::prepare_fresh_overlap(const std::string& solver)
{
#ifdef __ELPA
    if (solver == "genelpa")
    {
        hsolver::DiagoElpa<double>::DecomposedState = 0;
    }
    else if (solver == "elpa")
    {
        hsolver::DiagoElpaNative<double>::DecomposedState = 0;
    }
#else
    (void)solver;
#endif
}

} // namespace fde
