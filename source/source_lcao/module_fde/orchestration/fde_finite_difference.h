#ifndef FDE_FINITE_DIFFERENCE_H
#define FDE_FINITE_DIFFERENCE_H

#include "source_lcao/module_fde/orchestration/fde_pes_scan.h"

#include <string>

namespace fde
{

struct FiniteDifferenceStencil
{
    std::string coordinate_label;
    GeometryPoint reference;
    GeometryPoint minus_step;
    GeometryPoint minus_half_step;
    GeometryPoint plus_half_step;
    GeometryPoint plus_step;
};

struct FiniteDifferenceControls
{
    double step_bohr;
    double geometry_tolerance_bohr;
    double electron_tolerance;
    double minimum_localization_score;
    double maximum_force_difference_ry_per_bohr;
};

struct FiniteDifferenceForceResult
{
    std::string coordinate_label;
    std::string state_label;
    double minus_step_energy_ry;
    double minus_half_step_energy_ry;
    double plus_half_step_energy_ry;
    double plus_step_energy_ry;
    double coarse_force_ry_per_bohr;
    double fine_force_ry_per_bohr;
    double richardson_force_ry_per_bohr;
    double estimated_error_ry_per_bohr;
    int total_freeze_thaw_cycles;
};

class FiniteDifferenceForce
{
  public:
    static FiniteDifferenceForceResult evaluate(
        const FiniteDifferenceStencil& stencil,
        const DiabaticStateDefinition& state,
        const FreezeThawCheckpoint& reference_checkpoint,
        const FiniteDifferenceControls& controls,
        const DiabaticStateRunner& runner);
};

} // namespace fde

#endif // FDE_FINITE_DIFFERENCE_H
