#include "source_lcao/module_fde/embedding/fde_abacus_gamma_backend.h"

#include "source_base/matrix.h"
#include "source_hamilt/module_gint/gint_interface.h"

#include <limits>
#include <stdexcept>

namespace fde
{

void AbacusGintGammaIntegrator::potential_to_ao(
    const double* potential_ry,
    hamilt::HContainer<double>& matrix) const
{
    ModuleGint::cal_gint_vl(potential_ry, &matrix);
}

void AbacusGintGammaIntegrator::density_to_grid(
    const std::vector<hamilt::HContainer<double>*>& density_matrices,
    const int spin_channels,
    double** density_bohr3) const
{
    ModuleGint::cal_gint_rho(density_matrices, spin_channels, density_bohr3);
}

std::vector<double> AbacusGintGammaIntegrator::local_potential_force(
    const std::vector<const double*>& potential_ry,
    const std::vector<hamilt::HContainer<double>*>& density_matrices,
    const int spin_channels,
    const std::size_t atom_count) const
{
    if (atom_count == 0
        || atom_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("FDE Gint force atom count is invalid");
    }
    ModuleBase::matrix force(static_cast<int>(atom_count), 3);
    ModuleGint::cal_gint_fvl(spin_channels,
                             potential_ry,
                             density_matrices,
                             true,
                             false,
                             &force,
                             nullptr);
    std::vector<double> result(3 * atom_count, 0.0);
    for (std::size_t atom = 0; atom < atom_count; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            result[3 * atom + direction]
                = force(static_cast<int>(atom), static_cast<int>(direction));
        }
    }
    return result;
}

} // namespace fde
