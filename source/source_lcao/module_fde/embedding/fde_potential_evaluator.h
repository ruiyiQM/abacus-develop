#ifndef FDE_POTENTIAL_EVALUATOR_H
#define FDE_POTENTIAL_EVALUATOR_H

#include "source_lcao/module_fde/embedding/fde_semilocal_functional.h"

#include <vector>

namespace fde
{

class NonadditiveXcProvider
{
  public:
    virtual ~NonadditiveXcProvider() {}
    virtual FrozenSemilocalCache prepare_frozen(
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3,
        const GridDifferentialOperator* differential_operator) const = 0;
    virtual NonadditiveFunctionalResult evaluate(const SpinDensity& active,
                                                 const SpinDensity& frozen,
                                                 const UniformGrid& grid,
                                                 const double density_floor_bohr3,
                                                 const GridDifferentialOperator*
                                                     differential_operator = nullptr) const = 0;
    virtual NonadditiveFunctionalResult evaluate_cached(
        const SpinDensity& active,
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3,
        const FrozenSemilocalCache& frozen_cache,
        const GridDifferentialOperator* differential_operator) const = 0;
};

class DiracExchangeProvider : public NonadditiveXcProvider
{
  public:
    FrozenSemilocalCache prepare_frozen(
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3,
        const GridDifferentialOperator* differential_operator) const override;

    NonadditiveFunctionalResult evaluate(const SpinDensity& active,
                                         const SpinDensity& frozen,
                                         const UniformGrid& grid,
                                         const double density_floor_bohr3,
                                         const GridDifferentialOperator*
                                             differential_operator = nullptr) const override;

    NonadditiveFunctionalResult evaluate_cached(
        const SpinDensity& active,
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3,
        const FrozenSemilocalCache& frozen_cache,
        const GridDifferentialOperator* differential_operator) const override;
};

/** Spin-polarized PBE exchange-correlation evaluated by Libxc. */
class LibxcPbeProvider : public NonadditiveXcProvider
{
  public:
    static bool available();

    FrozenSemilocalCache prepare_frozen(
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3,
        const GridDifferentialOperator* differential_operator) const override;

    NonadditiveFunctionalResult evaluate(const SpinDensity& active,
                                         const SpinDensity& frozen,
                                         const UniformGrid& grid,
                                         const double density_floor_bohr3,
                                         const GridDifferentialOperator*
                                             differential_operator = nullptr) const override;

    NonadditiveFunctionalResult evaluate_cached(
        const SpinDensity& active,
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3,
        const FrozenSemilocalCache& frozen_cache,
        const GridDifferentialOperator* differential_operator) const override;
};

struct PotFdeConfig
{
    UniformGrid grid;
    KineticFunctional kinetic_functional;
    double density_floor_bohr3;
};

struct EmbeddingPotentialResult
{
    SpinPotential potential;
    double hartree_cross_energy_ry;
    double nonadditive_kinetic_energy_ry;
    double nonadditive_xc_energy_ry;
};

struct FrozenEmbeddingCache
{
    FrozenSemilocalCache kinetic;
    FrozenSemilocalCache xc;
};

class EmbeddingPotentialEvaluator
{
  public:
    static FrozenEmbeddingCache prepare_frozen(
        const SpinDensity& frozen_density,
        const PotFdeConfig& config,
        const NonadditiveXcProvider& xc_provider,
        const GridDifferentialOperator* differential_operator);

    static EmbeddingPotentialResult evaluate(
        const SpinDensity& active_density,
        const SpinDensity& frozen_density,
        const std::vector<double>& frozen_hartree_potential_ry,
        const PotFdeConfig& config,
        const NonadditiveXcProvider& xc_provider,
        const GridDifferentialOperator* differential_operator = nullptr);

    static EmbeddingPotentialResult evaluate_cached(
        const SpinDensity& active_density,
        const SpinDensity& frozen_density,
        const std::vector<double>& frozen_hartree_potential_ry,
        const PotFdeConfig& config,
        const NonadditiveXcProvider& xc_provider,
        const FrozenEmbeddingCache& frozen_cache,
        const GridDifferentialOperator* differential_operator);
};

} // namespace fde

#endif // FDE_POTENTIAL_EVALUATOR_H
