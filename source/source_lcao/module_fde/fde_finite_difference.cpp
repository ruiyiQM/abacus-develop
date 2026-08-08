#include "fde_finite_difference.h"

#include <cmath>
#include <stdexcept>

namespace fde
{

namespace
{

void validate_text(const std::string& value, const char* description)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(std::string("FDE finite-difference ") + description
                                    + " must be nonempty and contain no whitespace");
    }
}

void validate_controls(const FiniteDifferenceControls& controls)
{
    if (!std::isfinite(controls.step_bohr) || controls.step_bohr <= 0.0
        || !std::isfinite(controls.geometry_tolerance_bohr)
        || controls.geometry_tolerance_bohr < 0.0
        || !std::isfinite(controls.electron_tolerance)
        || controls.electron_tolerance < 0.0
        || !std::isfinite(controls.minimum_localization_score)
        || controls.minimum_localization_score < 0.0
        || controls.minimum_localization_score > 1.0
        || !std::isfinite(controls.maximum_force_difference_ry_per_bohr)
        || controls.maximum_force_difference_ry_per_bohr <= 0.0)
    {
        throw std::invalid_argument("FDE finite-difference controls are invalid");
    }
}

void validate_geometry(const GeometryPoint& geometry)
{
    validate_text(geometry.label, "geometry label");
    validate_text(geometry.geometry_fingerprint, "geometry fingerprint");
    if (!std::isfinite(geometry.coordinate_bohr))
    {
        throw std::invalid_argument("FDE finite-difference coordinate must be finite");
    }
}

void require_coordinate(const double actual,
                        const double expected,
                        const double tolerance)
{
    if (std::fabs(actual - expected) > tolerance)
    {
        throw std::invalid_argument("FDE finite-difference geometry does not match its stencil position");
    }
}

void validate_stencil(const FiniteDifferenceStencil& stencil,
                      const FiniteDifferenceControls& controls)
{
    validate_text(stencil.coordinate_label, "coordinate label");
    validate_geometry(stencil.reference);
    validate_geometry(stencil.minus_step);
    validate_geometry(stencil.minus_half_step);
    validate_geometry(stencil.plus_half_step);
    validate_geometry(stencil.plus_step);

    const double reference = stencil.reference.coordinate_bohr;
    const double step = controls.step_bohr;
    require_coordinate(stencil.minus_step.coordinate_bohr,
                       reference - step,
                       controls.geometry_tolerance_bohr);
    require_coordinate(stencil.minus_half_step.coordinate_bohr,
                       reference - 0.5 * step,
                       controls.geometry_tolerance_bohr);
    require_coordinate(stencil.plus_half_step.coordinate_bohr,
                       reference + 0.5 * step,
                       controls.geometry_tolerance_bohr);
    require_coordinate(stencil.plus_step.coordinate_bohr,
                       reference + step,
                       controls.geometry_tolerance_bohr);

    const GeometryPoint* points[5]
        = {&stencil.reference, &stencil.minus_step, &stencil.minus_half_step,
           &stencil.plus_half_step, &stencil.plus_step};
    for (int first = 0; first < 5; ++first)
    {
        for (int second = 0; second < first; ++second)
        {
            if (points[first]->label == points[second]->label
                || points[first]->geometry_fingerprint
                       == points[second]->geometry_fingerprint)
            {
                throw std::invalid_argument("FDE finite-difference geometries must have unique identities");
            }
        }
    }
}

void validate_state_identity(const FreezeThawCheckpoint& checkpoint,
                             const GeometryPoint& geometry,
                             const DiabaticStateDefinition& state,
                             const FiniteDifferenceControls& controls)
{
    DensityArtifactIO::validate_compatible_pair(checkpoint.first,
                                                checkpoint.second,
                                                controls.electron_tolerance);
    if (!checkpoint.converged || !checkpoint.first.scf_converged
        || !checkpoint.second.scf_converged)
    {
        throw std::runtime_error("FDE finite-difference displacement did not converge freeze-thaw");
    }
    if (checkpoint.first.state_label != state.label
        || checkpoint.second.state_label != state.label
        || checkpoint.first.geometry_fingerprint != geometry.geometry_fingerprint
        || checkpoint.second.geometry_fingerprint != geometry.geometry_fingerprint)
    {
        throw std::invalid_argument("FDE finite-difference result has the wrong state or geometry");
    }
    if (checkpoint.first.fragment_label != state.first_fragment_label
        || checkpoint.second.fragment_label != state.second_fragment_label
        || checkpoint.first.alpha_electrons != state.first_alpha_electrons
        || checkpoint.first.beta_electrons != state.first_beta_electrons
        || checkpoint.second.alpha_electrons != state.second_alpha_electrons
        || checkpoint.second.beta_electrons != state.second_beta_electrons)
    {
        throw std::invalid_argument("FDE finite-difference result changed diabatic populations");
    }
    if (checkpoint.completed_cycles != checkpoint.first.freeze_thaw_cycle
        || checkpoint.completed_cycles != checkpoint.second.freeze_thaw_cycle)
    {
        throw std::invalid_argument("FDE finite-difference result has inconsistent cycles");
    }
    CanonicalEnergy::validate(checkpoint.energy);
}

struct DisplacedEnergy
{
    double energy_ry;
    int cycles;
};

DisplacedEnergy run_displacement(const GeometryPoint& geometry,
                                 const DiabaticStateDefinition& state,
                                 const FreezeThawCheckpoint& reference_checkpoint,
                                 const FiniteDifferenceControls& controls,
                                 const DiabaticStateRunner& runner)
{
    const DiabaticStateRunResult result = runner.run(geometry,
                                                     state,
                                                     &reference_checkpoint);
    validate_state_identity(result.checkpoint, geometry, state, controls);
    if (!std::isfinite(result.localization_score)
        || result.localization_score < controls.minimum_localization_score
        || result.localization_score > 1.0)
    {
        throw std::runtime_error("FDE finite-difference displacement lost diabatic localization");
    }
    return {CanonicalEnergy::total(result.checkpoint.energy),
            result.checkpoint.completed_cycles};
}

} // namespace

FiniteDifferenceForceResult FiniteDifferenceForce::evaluate(
    const FiniteDifferenceStencil& stencil,
    const DiabaticStateDefinition& state,
    const FreezeThawCheckpoint& reference_checkpoint,
    const FiniteDifferenceControls& controls,
    const DiabaticStateRunner& runner)
{
    validate_controls(controls);
    validate_stencil(stencil, controls);
    validate_state_identity(reference_checkpoint, stencil.reference, state, controls);

    const DisplacedEnergy minus_step = run_displacement(stencil.minus_step,
                                                        state,
                                                        reference_checkpoint,
                                                        controls,
                                                        runner);
    const DisplacedEnergy plus_step = run_displacement(stencil.plus_step,
                                                       state,
                                                       reference_checkpoint,
                                                       controls,
                                                       runner);
    const DisplacedEnergy minus_half = run_displacement(stencil.minus_half_step,
                                                        state,
                                                        reference_checkpoint,
                                                        controls,
                                                        runner);
    const DisplacedEnergy plus_half = run_displacement(stencil.plus_half_step,
                                                       state,
                                                       reference_checkpoint,
                                                       controls,
                                                       runner);

    FiniteDifferenceForceResult result;
    result.coordinate_label = stencil.coordinate_label;
    result.state_label = state.label;
    result.minus_step_energy_ry = minus_step.energy_ry;
    result.minus_half_step_energy_ry = minus_half.energy_ry;
    result.plus_half_step_energy_ry = plus_half.energy_ry;
    result.plus_step_energy_ry = plus_step.energy_ry;
    result.coarse_force_ry_per_bohr
        = -(plus_step.energy_ry - minus_step.energy_ry) / (2.0 * controls.step_bohr);
    result.fine_force_ry_per_bohr
        = -(plus_half.energy_ry - minus_half.energy_ry) / controls.step_bohr;
    const double force_difference
        = std::fabs(result.fine_force_ry_per_bohr - result.coarse_force_ry_per_bohr);
    if (force_difference > controls.maximum_force_difference_ry_per_bohr)
    {
        throw std::runtime_error("FDE finite-difference force failed the step-halving check");
    }
    result.richardson_force_ry_per_bohr
        = (4.0 * result.fine_force_ry_per_bohr - result.coarse_force_ry_per_bohr) / 3.0;
    result.estimated_error_ry_per_bohr = force_difference / 3.0;
    result.total_freeze_thaw_cycles
        = minus_step.cycles + minus_half.cycles + plus_half.cycles + plus_step.cycles;
    return result;
}

} // namespace fde
