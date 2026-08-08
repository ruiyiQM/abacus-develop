#ifndef FDE_POTENTIAL_EVALUATOR_H
#define FDE_POTENTIAL_EVALUATOR_H

#include "fde_semilocal_functional.h"

#include <vector>

namespace fde
{

class NonadditiveXcProvider
{
  public:
    virtual ~NonadditiveXcProvider() {}
    virtual NonadditiveFunctionalResult evaluate(const SpinDensity& active,
                                                 const SpinDensity& frozen,
                                                 const UniformGrid& grid,
                                                 const double density_floor_bohr3) const = 0;
};

class DiracExchangeProvider : public NonadditiveXcProvider
{
  public:
    NonadditiveFunctionalResult evaluate(const SpinDensity& active,
                                         const SpinDensity& frozen,
                                         const UniformGrid& grid,
                                         const double density_floor_bohr3) const override;
};

/** Spin-polarized PBE exchange-correlation evaluated by Libxc. */
class LibxcPbeProvider : public NonadditiveXcProvider
{
  public:
    static bool available();

    NonadditiveFunctionalResult evaluate(const SpinDensity& active,
                                         const SpinDensity& frozen,
                                         const UniformGrid& grid,
                                         const double density_floor_bohr3) const override;
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

class EmbeddingPotentialEvaluator
{
  public:
    static EmbeddingPotentialResult evaluate(
        const SpinDensity& active_density,
        const SpinDensity& frozen_density,
        const std::vector<double>& frozen_hartree_potential_ry,
        const PotFdeConfig& config,
        const NonadditiveXcProvider& xc_provider);
};

} // namespace fde

#endif // FDE_POTENTIAL_EVALUATOR_H
