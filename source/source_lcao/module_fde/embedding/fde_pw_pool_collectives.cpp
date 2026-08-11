#include "source_lcao/module_fde/embedding/fde_pw_pool_collectives.h"

#include "source_basis/module_pw/pw_basis.h"

#include <limits>
#include <stdexcept>

namespace fde
{

namespace
{

void validate(const ModulePW::PW_Basis& basis)
{
    if (basis.poolnproc <= 0 || basis.poolrank < 0 || basis.poolrank >= basis.poolnproc)
    {
        throw std::invalid_argument("FDE received an invalid PW pool layout");
    }
#ifdef __MPI
    if (basis.pool_world == MPI_COMM_NULL)
    {
        throw std::invalid_argument("FDE PW collective requires an initialized communicator");
    }
#endif
}

} // namespace

void PwPoolCollectives::sum_in_place(double* values, const std::size_t count, const ModulePW::PW_Basis& basis)
{
    validate(basis);
    if ((count != 0 && values == nullptr) || count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("FDE PW sum buffer is invalid");
    }
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, values, static_cast<int>(count), MPI_DOUBLE, MPI_SUM, basis.pool_world);
#else
    (void)values;
    (void)count;
#endif
}

double PwPoolCollectives::maximum(const double value, const ModulePW::PW_Basis& basis)
{
    validate(basis);
    double result = value;
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_MAX, basis.pool_world);
#endif
    return result;
}

} // namespace fde
