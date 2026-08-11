#include "fde_warm_start.h"

#include <limits>
#include <stdexcept>

namespace fde
{

FdeWarmStartPlan FdeWarmStart::plan(const std::size_t request_index)
{
    if (request_index
        > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            "FDE session request index exceeds the ABACUS ionic-step range");
    }
    FdeWarmStartPlan result;
    result.ionic_step = static_cast<int>(request_index);
    result.reuse_ao_density_matrix = request_index > 0;
    result.reuse_orbitals = request_index > 0;
    result.mode = request_index == 0
                      ? "density_seed"
                      : "resident_ao_density_matrix_and_orbitals";
    return result;
}

} // namespace fde
