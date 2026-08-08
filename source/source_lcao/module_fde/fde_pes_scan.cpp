#include "fde_pes_scan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace fde
{

namespace
{

const int pes_schema_version = 1;

void validate_text(const std::string& value, const char* description)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(std::string("FDE PES ") + description
                                    + " must be nonempty and contain no whitespace");
    }
}

void validate_inputs(const std::vector<GeometryPoint>& geometries,
                     const std::vector<DiabaticStateDefinition>& states,
                     const PesScanControls& controls)
{
    if (geometries.empty())
    {
        throw std::invalid_argument("FDE PES scan requires at least one geometry");
    }
    if (states.size() != 2 || states[0].label == states[1].label)
    {
        throw std::invalid_argument("FDE PES scan requires exactly two distinct states");
    }
    if (!std::isfinite(controls.electron_tolerance) || controls.electron_tolerance < 0.0
        || !std::isfinite(controls.minimum_localization_score)
        || controls.minimum_localization_score < 0.0
        || controls.minimum_localization_score > 1.0)
    {
        throw std::invalid_argument("FDE PES scan controls are invalid");
    }

    for (std::size_t state_index = 0; state_index < states.size(); ++state_index)
    {
        const DiabaticStateDefinition& state = states[state_index];
        validate_text(state.label, "state label");
        validate_text(state.first_fragment_label, "first fragment label");
        validate_text(state.second_fragment_label, "second fragment label");
        if (state.first_fragment_label == state.second_fragment_label
            || state.first_alpha_electrons < 0 || state.first_beta_electrons < 0
            || state.second_alpha_electrons < 0 || state.second_beta_electrons < 0)
        {
            throw std::invalid_argument("FDE PES state fragment populations are invalid");
        }
    }

    for (std::size_t geometry_index = 0; geometry_index < geometries.size(); ++geometry_index)
    {
        const GeometryPoint& geometry = geometries[geometry_index];
        validate_text(geometry.label, "geometry label");
        validate_text(geometry.geometry_fingerprint, "geometry fingerprint");
        if (!std::isfinite(geometry.coordinate_bohr)
            || (geometry_index > 0
                && geometry.coordinate_bohr <= geometries[geometry_index - 1].coordinate_bohr))
        {
            throw std::invalid_argument("FDE PES geometry coordinates must be finite and increasing");
        }
        for (std::size_t previous = 0; previous < geometry_index; ++previous)
        {
            if (geometry.label == geometries[previous].label
                || geometry.geometry_fingerprint
                       == geometries[previous].geometry_fingerprint)
            {
                throw std::invalid_argument("FDE PES geometry identities must be unique");
            }
        }
    }
}

void validate_state_result(const DiabaticStateRunResult& result,
                           const GeometryPoint& geometry,
                           const DiabaticStateDefinition& state,
                           const PesScanControls& controls)
{
    const FreezeThawCheckpoint& checkpoint = result.checkpoint;
    DensityArtifactIO::validate_compatible_pair(checkpoint.first,
                                                checkpoint.second,
                                                controls.electron_tolerance);
    if (!checkpoint.converged || !checkpoint.first.scf_converged
        || !checkpoint.second.scf_converged)
    {
        throw std::runtime_error("FDE PES point did not converge its complete freeze-thaw workflow");
    }
    if (checkpoint.first.state_label != state.label
        || checkpoint.second.state_label != state.label
        || checkpoint.first.geometry_fingerprint != geometry.geometry_fingerprint
        || checkpoint.second.geometry_fingerprint != geometry.geometry_fingerprint)
    {
        throw std::invalid_argument("FDE PES runner returned the wrong state or geometry identity");
    }
    if (checkpoint.first.fragment_label != state.first_fragment_label
        || checkpoint.second.fragment_label != state.second_fragment_label
        || checkpoint.first.alpha_electrons != state.first_alpha_electrons
        || checkpoint.first.beta_electrons != state.first_beta_electrons
        || checkpoint.second.alpha_electrons != state.second_alpha_electrons
        || checkpoint.second.beta_electrons != state.second_beta_electrons)
    {
        throw std::invalid_argument("FDE PES runner changed the diabatic fragment populations");
    }
    if (checkpoint.completed_cycles != checkpoint.first.freeze_thaw_cycle
        || checkpoint.completed_cycles != checkpoint.second.freeze_thaw_cycle)
    {
        throw std::invalid_argument("FDE PES runner returned inconsistent cycle metadata");
    }
    CanonicalEnergy::validate(checkpoint.energy);
    if (!std::isfinite(result.localization_score)
        || result.localization_score < controls.minimum_localization_score
        || result.localization_score > 1.0)
    {
        throw std::runtime_error("FDE PES point failed the diabatic localization threshold");
    }
}

DiabaticPesDiagnostics diagnostics(const std::vector<GeometryPoint>& geometries,
                                   const std::vector<DiabaticPesPoint>& points)
{
    DiabaticPesDiagnostics result;
    result.minimum_localization_score = 1.0;
    result.minimum_absolute_gap_ry = std::numeric_limits<double>::infinity();
    result.maximum_adjacent_energy_change_ry = 0.0;
    result.maximum_slope_change_ry_per_bohr = 0.0;
    result.crossing_brackets = 0;

    for (std::size_t index = 0; index < points.size(); ++index)
    {
        result.minimum_localization_score
            = std::min(result.minimum_localization_score, points[index].localization_score);
    }
    for (std::size_t geometry_index = 0; geometry_index < geometries.size(); ++geometry_index)
    {
        const double gap = points[2 * geometry_index].energy_ry
                           - points[2 * geometry_index + 1].energy_ry;
        result.minimum_absolute_gap_ry
            = std::min(result.minimum_absolute_gap_ry, std::fabs(gap));
        if (geometry_index > 0)
        {
            const double previous_gap = points[2 * (geometry_index - 1)].energy_ry
                                        - points[2 * (geometry_index - 1) + 1].energy_ry;
            if ((previous_gap < 0.0 && gap >= 0.0)
                || (previous_gap > 0.0 && gap <= 0.0))
            {
                ++result.crossing_brackets;
            }
        }
    }

    for (std::size_t state_index = 0; state_index < 2; ++state_index)
    {
        double previous_slope = 0.0;
        bool have_previous_slope = false;
        for (std::size_t geometry_index = 1; geometry_index < geometries.size(); ++geometry_index)
        {
            const double energy_change
                = points[2 * geometry_index + state_index].energy_ry
                  - points[2 * (geometry_index - 1) + state_index].energy_ry;
            result.maximum_adjacent_energy_change_ry
                = std::max(result.maximum_adjacent_energy_change_ry,
                           std::fabs(energy_change));
            const double coordinate_change
                = geometries[geometry_index].coordinate_bohr
                  - geometries[geometry_index - 1].coordinate_bohr;
            const double slope = energy_change / coordinate_change;
            if (have_previous_slope)
            {
                result.maximum_slope_change_ry_per_bohr
                    = std::max(result.maximum_slope_change_ry_per_bohr,
                               std::fabs(slope - previous_slope));
            }
            previous_slope = slope;
            have_previous_slope = true;
        }
    }
    return result;
}

} // namespace

DiabaticPes TwoStatePesScan::run(
    const std::vector<GeometryPoint>& geometries,
    const std::vector<DiabaticStateDefinition>& states,
    const PesScanControls& controls,
    const DiabaticStateRunner& runner)
{
    validate_inputs(geometries, states, controls);
    DiabaticPes pes;
    pes.schema_version = pes_schema_version;
    pes.first_state_label = states[0].label;
    pes.second_state_label = states[1].label;

    FreezeThawCheckpoint warm_starts[2];
    bool have_warm_start[2] = {false, false};
    for (std::size_t geometry_index = 0; geometry_index < geometries.size(); ++geometry_index)
    {
        for (std::size_t state_index = 0; state_index < states.size(); ++state_index)
        {
            const FreezeThawCheckpoint* warm_start
                = have_warm_start[state_index] ? &warm_starts[state_index] : 0;
            const DiabaticStateRunResult run_result
                = runner.run(geometries[geometry_index], states[state_index], warm_start);
            validate_state_result(run_result,
                                  geometries[geometry_index],
                                  states[state_index],
                                  controls);
            warm_starts[state_index] = run_result.checkpoint;
            have_warm_start[state_index] = true;

            DiabaticPesPoint point;
            point.geometry_label = geometries[geometry_index].label;
            point.geometry_fingerprint = geometries[geometry_index].geometry_fingerprint;
            point.coordinate_bohr = geometries[geometry_index].coordinate_bohr;
            point.state_label = states[state_index].label;
            point.energy_ry = CanonicalEnergy::total(run_result.checkpoint.energy);
            point.localization_score = run_result.localization_score;
            point.freeze_thaw_cycles = run_result.checkpoint.completed_cycles;
            pes.points.push_back(point);
        }
    }
    pes.diagnostics = diagnostics(geometries, pes.points);
    return pes;
}

void TwoStatePesScan::write(std::ostream& output, const DiabaticPes& pes)
{
    if (pes.schema_version != pes_schema_version || pes.points.empty())
    {
        throw std::invalid_argument("FDE PES output is empty or has an unsupported schema");
    }
    output.precision(17);
    output << "FDE_DIABATIC_PES " << pes.schema_version << '\n';
    output << "STATES " << pes.first_state_label << ' ' << pes.second_state_label << '\n';
    output << "COLUMNS geometry fingerprint coordinate_bohr state energy_ry localization cycles\n";
    for (std::size_t index = 0; index < pes.points.size(); ++index)
    {
        const DiabaticPesPoint& point = pes.points[index];
        output << "POINT " << point.geometry_label << ' ' << point.geometry_fingerprint << ' '
               << point.coordinate_bohr << ' ' << point.state_label << ' '
               << point.energy_ry << ' ' << point.localization_score << ' '
               << point.freeze_thaw_cycles << '\n';
    }
    output << "DIAGNOSTICS " << pes.diagnostics.minimum_localization_score << ' '
           << pes.diagnostics.minimum_absolute_gap_ry << ' '
           << pes.diagnostics.maximum_adjacent_energy_change_ry << ' '
           << pes.diagnostics.maximum_slope_change_ry_per_bohr << ' '
           << pes.diagnostics.crossing_brackets << '\n';
    output << "END\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE diabatic PES");
    }
}

} // namespace fde
