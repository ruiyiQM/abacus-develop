#ifndef FDE_FREEZE_THAW_H
#define FDE_FREEZE_THAW_H

#include "source_lcao/module_fde/io/fde_density_artifact.h"
#include "source_lcao/module_fde/orchestration/fde_energy_ledger.h"

#include <iosfwd>

namespace fde
{

enum class FreezeThawOrder
{
    FirstThenSecond,
    SecondThenFirst
};

struct FreezeThawControls
{
    int maximum_cycles;
    double density_tolerance;
    double energy_tolerance_ry;
    double electron_tolerance;
    FreezeThawOrder order;
};

struct FreezeThawCheckpoint
{
    int schema_version;
    FrozenDensityArtifact first;
    FrozenDensityArtifact second;
    EnergyLedger energy;
    int completed_cycles;
    bool converged;
    double density_residual;
    double energy_change_ry;
};

class FreezeThawStepRunner
{
  public:
    virtual ~FreezeThawStepRunner() {}

    virtual FrozenDensityArtifact update(
        const FrozenDensityArtifact& active,
        const FrozenDensityArtifact& frozen) const = 0;

    virtual EnergyLedger canonical_energy(
        const FrozenDensityArtifact& first,
        const FrozenDensityArtifact& second) const = 0;
};

class FreezeThawWorkflow
{
  public:
    static FreezeThawCheckpoint initialize(
        const FrozenDensityArtifact& first,
        const FrozenDensityArtifact& second,
        const FreezeThawStepRunner& runner,
        double electron_tolerance);

    static FreezeThawCheckpoint run(
        const FreezeThawCheckpoint& checkpoint,
        const FreezeThawControls& controls,
        const FreezeThawStepRunner& runner);

    static void write_checkpoint(
        std::ostream& output,
        const FreezeThawCheckpoint& checkpoint);

    static FreezeThawCheckpoint read_checkpoint(
        std::istream& input,
        double electron_tolerance);
};

} // namespace fde

#endif // FDE_FREEZE_THAW_H
