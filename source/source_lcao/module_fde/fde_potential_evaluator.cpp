#include "fde_potential_evaluator.h"

#include <stdexcept>

namespace fde
{

NonadditiveFunctionalResult DiracExchangeProvider::evaluate(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3) const
{
    return SemilocalFunctional::nonadditive_dirac_exchange(active,
                                                           frozen,
                                                           grid,
                                                           density_floor_bohr3);
}

EmbeddingPotentialResult EmbeddingPotentialEvaluator::evaluate(
    const SpinDensity& active_density,
    const SpinDensity& frozen_density,
    const std::vector<double>& frozen_hartree_potential_ry,
    const PotFdeConfig& config,
    const NonadditiveXcProvider& xc_provider)
{
    const NonadditiveFunctionalResult kinetic
        = SemilocalFunctional::nonadditive_kinetic(active_density,
                                                   frozen_density,
                                                   config.grid,
                                                   config.kinetic_functional,
                                                   config.density_floor_bohr3);
    const NonadditiveFunctionalResult xc
        = xc_provider.evaluate(active_density,
                               frozen_density,
                               config.grid,
                               config.density_floor_bohr3);
    const std::size_t size = frozen_hartree_potential_ry.size();
    if (active_density.alpha_bohr3.size() != size || active_density.beta_bohr3.size() != size)
    {
        throw std::invalid_argument("FDE active density does not match the frozen potential grid");
    }

    EmbeddingPotentialResult result;
    result.potential.alpha_ry.resize(size);
    result.potential.beta_ry.resize(size);
    result.hartree_cross_energy_ry = 0.0;
    result.nonadditive_kinetic_energy_ry = kinetic.energy_ry;
    result.nonadditive_xc_energy_ry = xc.energy_ry;
    const double volume_element = config.grid.spacing_x_bohr
                                  * config.grid.spacing_y_bohr
                                  * config.grid.spacing_z_bohr;
    for (std::size_t index = 0; index < size; ++index)
    {
        result.potential.alpha_ry[index]
            = frozen_hartree_potential_ry[index]
              + kinetic.active_potential.alpha_ry[index]
              + xc.active_potential.alpha_ry[index];
        result.potential.beta_ry[index]
            = frozen_hartree_potential_ry[index]
              + kinetic.active_potential.beta_ry[index]
              + xc.active_potential.beta_ry[index];
        result.hartree_cross_energy_ry
            += (active_density.alpha_bohr3[index] + active_density.beta_bohr3[index])
               * frozen_hartree_potential_ry[index] * volume_element;
    }
    return result;
}

} // namespace fde
