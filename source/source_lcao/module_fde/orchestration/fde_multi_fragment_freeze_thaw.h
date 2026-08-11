#ifndef FDE_MULTI_FRAGMENT_FREEZE_THAW_H
#define FDE_MULTI_FRAGMENT_FREEZE_THAW_H

#include "source_lcao/module_fde/io/fde_density_artifact.h"
#include "source_lcao/module_fde/orchestration/fde_multi_fragment_energy.h"

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace fde
{

struct MultiFragmentFreezeThawControls
{
    int maximum_cycles;
    double density_tolerance;
    double energy_tolerance_ry;
    double electron_tolerance;
    std::vector<std::size_t> update_order;
};

struct MultiFragmentFreezeThawCheckpoint
{
    int schema_version;
    std::vector<FrozenDensityArtifact> fragments;
    MultiFragmentEnergyLedger energy;
    int completed_cycles;
    bool converged;
    double density_residual;
    double energy_change_ry;
};

class MultiFragmentFreezeThawStepRunner
{
  public:
    virtual ~MultiFragmentFreezeThawStepRunner() {}

    virtual FrozenDensityArtifact update(
        const FrozenDensityArtifact& active,
        const std::vector<FrozenDensityArtifact>& frozen_environment) const = 0;

    virtual MultiFragmentEnergyLedger canonical_energy(
        const std::vector<FrozenDensityArtifact>& fragments) const = 0;
};

class MultiFragmentFreezeThawWorkflow
{
  public:
    static MultiFragmentFreezeThawCheckpoint initialize(
        const std::vector<FrozenDensityArtifact>& fragments,
        const MultiFragmentFreezeThawStepRunner& runner,
        double electron_tolerance);

    static MultiFragmentFreezeThawCheckpoint run(
        const MultiFragmentFreezeThawCheckpoint& checkpoint,
        const MultiFragmentFreezeThawControls& controls,
        const MultiFragmentFreezeThawStepRunner& runner);

    static void write_checkpoint(
        std::ostream& output,
        const MultiFragmentFreezeThawCheckpoint& checkpoint);

    static MultiFragmentFreezeThawCheckpoint read_checkpoint(
        std::istream& input,
        double electron_tolerance);
};

} // namespace fde

#endif // FDE_MULTI_FRAGMENT_FREEZE_THAW_H
