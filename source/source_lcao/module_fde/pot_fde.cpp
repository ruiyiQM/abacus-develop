#include "pot_fde.h"

#include <cmath>
#include <stdexcept>

namespace fde
{

PotFde::PotFde(const ModulePW::PW_Basis* rho_basis,
               const SpinDensity& frozen_density,
               const std::vector<double>& frozen_hartree_potential_ry,
               const PotFdeConfig& config,
               const std::shared_ptr<const NonadditiveXcProvider>& xc_provider)
    : frozen_density_(frozen_density),
      frozen_hartree_potential_ry_(frozen_hartree_potential_ry),
      config_(config),
      xc_provider_(xc_provider)
{
    this->rho_basis_ = rho_basis;
    this->dynamic_mode = true;
    this->fixed_mode = false;
    if (rho_basis == nullptr || xc_provider_ == nullptr)
    {
        throw std::invalid_argument("FDE potential requires a density basis and XC provider");
    }
    const std::size_t size = config_.grid.x * config_.grid.y * config_.grid.z;
    if (size != frozen_density_.alpha_bohr3.size()
        || size != frozen_density_.beta_bohr3.size()
        || size != frozen_hartree_potential_ry_.size()
        || static_cast<std::size_t>(rho_basis->nrxx) != size)
    {
        throw std::invalid_argument("FDE potential data must match the replicated ABACUS density grid");
    }
    last_result_.potential.alpha_ry.assign(size, 0.0);
    last_result_.potential.beta_ry.assign(size, 0.0);
    last_result_.hartree_cross_energy_ry = 0.0;
    last_result_.nonadditive_kinetic_energy_ry = 0.0;
    last_result_.nonadditive_xc_energy_ry = 0.0;
}

EmbeddingPotentialResult PotFde::evaluate(const SpinDensity& active_density) const
{
    return EmbeddingPotentialEvaluator::evaluate(active_density,
                                                 frozen_density_,
                                                 frozen_hartree_potential_ry_,
                                                 config_,
                                                 *xc_provider_);
}

void PotFde::cal_v_eff(const Charge* const charge,
                       const UnitCell* const unit_cell,
                       ModuleBase::matrix& effective_potential)
{
    (void)unit_cell;
    if (charge == nullptr || charge->rho == nullptr || charge->nspin != 2
        || effective_potential.nr != 2
        || static_cast<std::size_t>(effective_potential.nc)
               != frozen_hartree_potential_ry_.size())
    {
        throw std::invalid_argument("FDE potential requires a matching collinear-spin Charge object");
    }
    SpinDensity active;
    active.alpha_bohr3.assign(charge->rho[0], charge->rho[0] + effective_potential.nc);
    active.beta_bohr3.assign(charge->rho[1], charge->rho[1] + effective_potential.nc);
    last_result_ = this->evaluate(active);
    for (int index = 0; index < effective_potential.nc; ++index)
    {
        effective_potential(0, index) += last_result_.potential.alpha_ry[index];
        effective_potential(1, index) += last_result_.potential.beta_ry[index];
    }
}

double PotFde::get_energy() const
{
    return last_result_.hartree_cross_energy_ry
           + last_result_.nonadditive_kinetic_energy_ry
           + last_result_.nonadditive_xc_energy_ry;
}

const EmbeddingPotentialResult& PotFde::last_result() const
{
    return last_result_;
}

} // namespace fde
