#ifndef POT_FDE_H
#define POT_FDE_H

#include "source_lcao/module_fde/embedding/fde_potential_evaluator.h"
#include "source_lcao/module_fde/embedding/fde_pw_grid_differential.h"
#include "source_estate/module_pot/pot_base.h"

#include <memory>
#include <vector>

namespace fde
{

class PotFde : public elecstate::PotBase
{
  public:
    PotFde(const ModulePW::PW_Basis* rho_basis,
           const SpinDensity& frozen_density,
           const std::vector<double>& frozen_hartree_potential_ry,
           const PotFdeConfig& config,
           const std::shared_ptr<const NonadditiveXcProvider>& xc_provider,
           bool use_gpu);

    EmbeddingPotentialResult evaluate(const SpinDensity& active_density) const;

    /** Replace the per-request frozen environment without reallocating FFT state. */
    void reset_frozen_density(
        const SpinDensity& frozen_density,
        const std::vector<double>& frozen_hartree_potential_ry,
        const PotFdeConfig& config);

    void cal_v_eff(const Charge* const charge,
                   const UnitCell* const unit_cell,
                   ModuleBase::matrix& effective_potential) override;

    double get_energy() const override;
    const EmbeddingPotentialResult& last_result() const;

  private:
    SpinDensity frozen_density_;
    std::vector<double> frozen_hartree_potential_ry_;
    PotFdeConfig config_;
    std::shared_ptr<const NonadditiveXcProvider> xc_provider_;
    std::unique_ptr<PwGridDifferential> differential_operator_;
    FrozenEmbeddingCache frozen_cache_;
    EmbeddingPotentialResult last_result_;

    void validate_frozen_data(
        const SpinDensity& frozen_density,
        const std::vector<double>& frozen_hartree_potential_ry,
        const PotFdeConfig& config) const;
    void reset_last_result();
};

} // namespace fde

#endif // POT_FDE_H
