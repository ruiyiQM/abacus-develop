#include "fde_abacus_gamma_backend.h"

#include "source_hamilt/module_gint/gint_interface.h"

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

} // namespace fde
