#include "pot_fde.h"

#include "fde_pw_pool_collectives.h"
#include "source_base/timer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fde
{

namespace
{

class ScopedPotFdeTimer
{
  public:
    explicit ScopedPotFdeTimer(const std::string& name) : name_(name)
    {
        ModuleBase::timer::start("PotFde", name_);
    }

    ~ScopedPotFdeTimer()
    {
        ModuleBase::timer::end("PotFde", name_);
    }

  private:
    std::string name_;
};

double sanitize_charge_density(const double value)
{
    // AO-density projection can ring below zero around density nodes.  Apply
    // the same nonnegative projection used by FdeLcaoDriver when normalizing
    // a converged density. NaN/Inf validation is performed before entering
    // the OpenMP loop because exceptions must not escape a parallel region.
    return std::max(0.0, value);
}

} // namespace

PotFde::PotFde(const ModulePW::PW_Basis* rho_basis,
               const SpinDensity& frozen_density,
               const std::vector<double>& frozen_hartree_potential_ry,
               const PotFdeConfig& config,
               const std::shared_ptr<const NonadditiveXcProvider>& xc_provider,
               const bool use_gpu)
    : frozen_density_(frozen_density),
      frozen_hartree_potential_ry_(frozen_hartree_potential_ry),
      config_(config),
      xc_provider_(xc_provider),
      differential_operator_()
{
    this->rho_basis_ = rho_basis;
    this->dynamic_mode = true;
    this->fixed_mode = false;
    if (rho_basis == nullptr || xc_provider_ == nullptr)
    {
        throw std::invalid_argument("FDE potential requires a density basis and XC provider");
    }
    this->validate_frozen_data(frozen_density_,
                               frozen_hartree_potential_ry_,
                               config_);
    differential_operator_.reset(new PwGridDifferential(*rho_basis, use_gpu));
    {
        const ScopedPotFdeTimer cache_timer("prepare_frozen_cache");
        frozen_cache_ = EmbeddingPotentialEvaluator::prepare_frozen(
            frozen_density_, config_, *xc_provider_, differential_operator_.get());
    }
    this->reset_last_result();
}

void PotFde::validate_frozen_data(
    const SpinDensity& frozen_density,
    const std::vector<double>& frozen_hartree_potential_ry,
    const PotFdeConfig& config) const
{
    if (this->rho_basis_ == nullptr
        || config.grid.x != static_cast<std::size_t>(this->rho_basis_->nx)
        || config.grid.y != static_cast<std::size_t>(this->rho_basis_->ny)
        || config.grid.z != static_cast<std::size_t>(this->rho_basis_->nz))
    {
        throw std::invalid_argument(
            "FDE potential grid does not match PW_Basis dimensions");
    }
    const std::size_t size
        = static_cast<std::size_t>(this->rho_basis_->nrxx);
    if (size != frozen_density.alpha_bohr3.size()
        || size != frozen_density.beta_bohr3.size()
        || size != frozen_hartree_potential_ry.size())
    {
        throw std::invalid_argument(
            "FDE potential data must match the local ABACUS density slab");
    }
}

void PotFde::reset_last_result()
{
    const std::size_t size
        = static_cast<std::size_t>(this->rho_basis_->nrxx);
    last_result_.potential.alpha_ry.assign(size, 0.0);
    last_result_.potential.beta_ry.assign(size, 0.0);
    last_result_.hartree_cross_energy_ry = 0.0;
    last_result_.nonadditive_kinetic_energy_ry = 0.0;
    last_result_.nonadditive_xc_energy_ry = 0.0;
}

void PotFde::reset_frozen_density(
    const SpinDensity& frozen_density,
    const std::vector<double>& frozen_hartree_potential_ry,
    const PotFdeConfig& config)
{
    this->validate_frozen_data(frozen_density,
                               frozen_hartree_potential_ry,
                               config);
    frozen_density_ = frozen_density;
    frozen_hartree_potential_ry_ = frozen_hartree_potential_ry;
    config_ = config;
    {
        const ScopedPotFdeTimer cache_timer("prepare_frozen_cache");
        frozen_cache_ = EmbeddingPotentialEvaluator::prepare_frozen(
            frozen_density_,
            config_,
            *xc_provider_,
            differential_operator_.get());
    }
    this->reset_last_result();
}

EmbeddingPotentialResult PotFde::evaluate(const SpinDensity& active_density) const
{
    const ScopedPotFdeTimer evaluate_timer("evaluate_functionals");
    EmbeddingPotentialResult result
        = EmbeddingPotentialEvaluator::evaluate_cached(active_density,
                                                       frozen_density_,
                                                       frozen_hartree_potential_ry_,
                                                       config_,
                                                       *xc_provider_,
                                                       frozen_cache_,
                                                       differential_operator_.get());
    double energies[3] = {result.hartree_cross_energy_ry,
                          result.nonadditive_kinetic_energy_ry,
                          result.nonadditive_xc_energy_ry};
    {
        const ScopedPotFdeTimer reduction_timer("reduce_energies");
        PwPoolCollectives::sum_in_place(energies, 3, *this->rho_basis_);
    }
    result.hartree_cross_energy_ry = energies[0];
    result.nonadditive_kinetic_energy_ry = energies[1];
    result.nonadditive_xc_energy_ry = energies[2];
    return result;
}

void PotFde::cal_v_eff(const Charge* const charge,
                       const UnitCell* const unit_cell,
                       ModuleBase::matrix& effective_potential)
{
    const ScopedPotFdeTimer total_timer("cal_v_eff");
    (void)unit_cell;
    if (charge == nullptr || charge->rho == nullptr
        || (charge->nspin != 1 && charge->nspin != 2)
        || effective_potential.nr != charge->nspin
        || static_cast<std::size_t>(effective_potential.nc)
               != frozen_hartree_potential_ry_.size())
    {
        throw std::invalid_argument(
            "FDE potential requires a matching RKS or collinear UKS Charge object");
    }
    {
        const ScopedPotFdeTimer validation_timer("validate_charge");
        for (int index = 0; index < effective_potential.nc; ++index)
        {
            if (!std::isfinite(charge->rho[0][index])
                || (charge->nspin == 2 && !std::isfinite(charge->rho[1][index])))
            {
                throw std::invalid_argument("FDE Charge density contains a non-finite value");
            }
        }
    }
    SpinDensity active;
    active.alpha_bohr3.resize(static_cast<std::size_t>(effective_potential.nc));
    active.beta_bohr3.resize(static_cast<std::size_t>(effective_potential.nc));
    {
        const ScopedPotFdeTimer density_timer("prepare_active_density");
#pragma omp parallel for schedule(static)
        for (int index = 0; index < effective_potential.nc; ++index)
        {
            if (charge->nspin == 1)
            {
                const double half_total
                    = 0.5 * sanitize_charge_density(charge->rho[0][index]);
                active.alpha_bohr3[index] = half_total;
                active.beta_bohr3[index] = half_total;
            }
            else
            {
                active.alpha_bohr3[index]
                    = sanitize_charge_density(charge->rho[0][index]);
                active.beta_bohr3[index]
                    = sanitize_charge_density(charge->rho[1][index]);
            }
        }
    }
    last_result_ = this->evaluate(active);
    {
        const ScopedPotFdeTimer assembly_timer("assemble_effective_potential");
#pragma omp parallel for schedule(static)
        for (int index = 0; index < effective_potential.nc; ++index)
        {
            if (charge->nspin == 1)
            {
                // Along the closed-shell constraint n_alpha = n_beta = n / 2,
                // dE/dn is one half of the sum of the two spin derivatives.
                effective_potential(0, index)
                    += 0.5 * (last_result_.potential.alpha_ry[index]
                              + last_result_.potential.beta_ry[index]);
            }
            else
            {
                effective_potential(0, index) += last_result_.potential.alpha_ry[index];
                effective_potential(1, index) += last_result_.potential.beta_ry[index];
            }
        }
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
