#include "source_lcao/module_fde/orchestration/fde_freeze_thaw.h"

#include <cmath>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fde
{

namespace
{

const int checkpoint_schema_version = 1;

void validate_controls(const FreezeThawControls& controls)
{
    if (controls.maximum_cycles <= 0
        || !std::isfinite(controls.density_tolerance)
        || controls.density_tolerance <= 0.0
        || !std::isfinite(controls.energy_tolerance_ry)
        || controls.energy_tolerance_ry <= 0.0
        || !std::isfinite(controls.electron_tolerance)
        || controls.electron_tolerance < 0.0
        || (controls.order != FreezeThawOrder::FirstThenSecond
            && controls.order != FreezeThawOrder::SecondThenFirst))
    {
        throw std::invalid_argument("FDE freeze-thaw controls are invalid");
    }
}

void validate_checkpoint(const FreezeThawCheckpoint& checkpoint,
                         const double electron_tolerance)
{
    if (checkpoint.schema_version != checkpoint_schema_version)
    {
        throw std::invalid_argument("Unsupported FDE freeze-thaw checkpoint schema version");
    }
    DensityArtifactIO::validate_compatible_pair(checkpoint.first,
                                                checkpoint.second,
                                                electron_tolerance);
    if (!checkpoint.first.scf_converged || !checkpoint.second.scf_converged)
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint contains an unconverged subsystem");
    }
    if (checkpoint.completed_cycles < 0
        || checkpoint.first.freeze_thaw_cycle != checkpoint.completed_cycles
        || checkpoint.second.freeze_thaw_cycle != checkpoint.completed_cycles)
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint cycle metadata is inconsistent");
    }
    if (!std::isfinite(checkpoint.density_residual) || checkpoint.density_residual < 0.0
        || !std::isfinite(checkpoint.energy_change_ry) || checkpoint.energy_change_ry < 0.0)
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint residuals are invalid");
    }
    CanonicalEnergy::validate(checkpoint.energy);
}

FrozenDensityArtifact update_fragment(const FrozenDensityArtifact& active,
                                      const FrozenDensityArtifact& frozen,
                                      const FreezeThawStepRunner& runner,
                                      const double electron_tolerance)
{
    const FrozenDensityArtifact updated = runner.update(active, frozen);
    DensityArtifactIO::validate_compatible_pair(updated, frozen, electron_tolerance);
    if (updated.fragment_label != active.fragment_label
        || updated.state_label != active.state_label
        || updated.alpha_electrons != active.alpha_electrons
        || updated.beta_electrons != active.beta_electrons)
    {
        throw std::invalid_argument("FDE freeze-thaw update changed subsystem identity or populations");
    }
    if (!updated.scf_converged)
    {
        throw std::runtime_error("FDE freeze-thaw update did not converge its one-way SCF");
    }
    if (updated.freeze_thaw_cycle != active.freeze_thaw_cycle + 1)
    {
        throw std::invalid_argument("FDE freeze-thaw update returned an unexpected cycle number");
    }
    return updated;
}

double pair_density_residual(const FrozenDensityArtifact& old_first,
                             const FrozenDensityArtifact& old_second,
                             const FrozenDensityArtifact& new_first,
                             const FrozenDensityArtifact& new_second)
{
    const std::size_t size = old_first.rho_alpha_bohr3.size();
    const double volume_element
        = old_first.cell_volume_bohr3 / static_cast<double>(size);
    double squared_residual = 0.0;
    for (std::size_t index = 0; index < size; ++index)
    {
        const double differences[4]
            = {new_first.rho_alpha_bohr3[index] - old_first.rho_alpha_bohr3[index],
               new_first.rho_beta_bohr3[index] - old_first.rho_beta_bohr3[index],
               new_second.rho_alpha_bohr3[index] - old_second.rho_alpha_bohr3[index],
               new_second.rho_beta_bohr3[index] - old_second.rho_beta_bohr3[index]};
        for (int spin_fragment = 0; spin_fragment < 4; ++spin_fragment)
        {
            squared_residual += differences[spin_fragment] * differences[spin_fragment]
                                * volume_element;
        }
    }
    return std::sqrt(squared_residual);
}

void require_token(std::istream& input, const char* expected)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        throw std::invalid_argument(std::string("FDE freeze-thaw checkpoint expected token ")
                                    + expected);
    }
}

void write_block(std::ostream& output, const char* label, const std::string& content)
{
    output << label << ' ' << content.size() << '\n';
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    output << '\n';
}

std::string read_block(std::istream& input, const char* label)
{
    require_token(input, label);
    std::size_t size = 0;
    input >> size;
    if (!input || size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint block size is invalid");
    }
    char separator = '\0';
    input.get(separator);
    if (separator != '\n')
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint block header is malformed");
    }
    std::string content(size, '\0');
    if (size > 0)
    {
        input.read(&content[0], static_cast<std::streamsize>(size));
    }
    input.get(separator);
    if (!input || separator != '\n')
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint block is truncated");
    }
    return content;
}

} // namespace

FreezeThawCheckpoint FreezeThawWorkflow::initialize(
    const FrozenDensityArtifact& first,
    const FrozenDensityArtifact& second,
    const FreezeThawStepRunner& runner,
    const double electron_tolerance)
{
    DensityArtifactIO::validate_compatible_pair(first, second, electron_tolerance);
    if (!first.scf_converged || !second.scf_converged
        || first.freeze_thaw_cycle != second.freeze_thaw_cycle)
    {
        throw std::invalid_argument("FDE freeze-thaw initialization requires matching converged cycles");
    }
    FreezeThawCheckpoint checkpoint;
    checkpoint.schema_version = checkpoint_schema_version;
    checkpoint.first = first;
    checkpoint.second = second;
    checkpoint.energy = runner.canonical_energy(first, second);
    CanonicalEnergy::validate(checkpoint.energy);
    checkpoint.completed_cycles = first.freeze_thaw_cycle;
    checkpoint.converged = false;
    checkpoint.density_residual = 0.0;
    checkpoint.energy_change_ry = 0.0;
    return checkpoint;
}

FreezeThawCheckpoint FreezeThawWorkflow::run(
    const FreezeThawCheckpoint& checkpoint,
    const FreezeThawControls& controls,
    const FreezeThawStepRunner& runner)
{
    validate_controls(controls);
    validate_checkpoint(checkpoint, controls.electron_tolerance);
    if (checkpoint.converged)
    {
        return checkpoint;
    }
    if (checkpoint.completed_cycles >= controls.maximum_cycles)
    {
        throw std::invalid_argument("FDE freeze-thaw maximum cycle is not beyond the checkpoint");
    }

    FreezeThawCheckpoint result = checkpoint;
    while (result.completed_cycles < controls.maximum_cycles && !result.converged)
    {
        const FrozenDensityArtifact old_first = result.first;
        const FrozenDensityArtifact old_second = result.second;
        const double old_energy = CanonicalEnergy::total(result.energy);

        if (controls.order == FreezeThawOrder::FirstThenSecond)
        {
            result.first = update_fragment(result.first,
                                           result.second,
                                           runner,
                                           controls.electron_tolerance);
            result.second = update_fragment(result.second,
                                            result.first,
                                            runner,
                                            controls.electron_tolerance);
        }
        else
        {
            result.second = update_fragment(result.second,
                                            result.first,
                                            runner,
                                            controls.electron_tolerance);
            result.first = update_fragment(result.first,
                                           result.second,
                                           runner,
                                           controls.electron_tolerance);
        }

        ++result.completed_cycles;
        result.density_residual = pair_density_residual(old_first,
                                                        old_second,
                                                        result.first,
                                                        result.second);
        result.energy = runner.canonical_energy(result.first, result.second);
        const double new_energy = CanonicalEnergy::total(result.energy);
        result.energy_change_ry = std::fabs(new_energy - old_energy);
        result.converged = result.density_residual <= controls.density_tolerance
                           && result.energy_change_ry <= controls.energy_tolerance_ry;
    }

    validate_checkpoint(result, controls.electron_tolerance);
    return result;
}

void FreezeThawWorkflow::write_checkpoint(
    std::ostream& output,
    const FreezeThawCheckpoint& checkpoint)
{
    validate_checkpoint(checkpoint, 1.0e-8);
    std::ostringstream first_output;
    std::ostringstream second_output;
    std::ostringstream energy_output;
    DensityArtifactIO::write(first_output, checkpoint.first);
    DensityArtifactIO::write(second_output, checkpoint.second);
    CanonicalEnergy::write(energy_output, checkpoint.energy);

    output.precision(17);
    output << "FDE_FREEZE_THAW_CHECKPOINT " << checkpoint.schema_version << '\n';
    output << "STATUS " << checkpoint.completed_cycles << ' '
           << (checkpoint.converged ? 1 : 0) << ' '
           << checkpoint.density_residual << ' '
           << checkpoint.energy_change_ry << '\n';
    write_block(output, "FIRST_ARTIFACT_BYTES", first_output.str());
    write_block(output, "SECOND_ARTIFACT_BYTES", second_output.str());
    write_block(output, "ENERGY_LEDGER_BYTES", energy_output.str());
    output << "END\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE freeze-thaw checkpoint");
    }
}

FreezeThawCheckpoint FreezeThawWorkflow::read_checkpoint(
    std::istream& input,
    const double electron_tolerance)
{
    FreezeThawCheckpoint checkpoint;
    require_token(input, "FDE_FREEZE_THAW_CHECKPOINT");
    input >> checkpoint.schema_version;
    require_token(input, "STATUS");
    int converged = 0;
    input >> checkpoint.completed_cycles >> converged
          >> checkpoint.density_residual >> checkpoint.energy_change_ry;
    if (converged != 0 && converged != 1)
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint convergence flag is invalid");
    }
    checkpoint.converged = converged == 1;

    const std::string first_content = read_block(input, "FIRST_ARTIFACT_BYTES");
    const std::string second_content = read_block(input, "SECOND_ARTIFACT_BYTES");
    const std::string energy_content = read_block(input, "ENERGY_LEDGER_BYTES");
    require_token(input, "END");
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE freeze-thaw checkpoint contains trailing content");
    }

    std::istringstream first_input(first_content);
    std::istringstream second_input(second_content);
    std::istringstream energy_input(energy_content);
    checkpoint.first = DensityArtifactIO::read(first_input, electron_tolerance);
    checkpoint.second = DensityArtifactIO::read(second_input, electron_tolerance);
    checkpoint.energy = CanonicalEnergy::read(energy_input);
    validate_checkpoint(checkpoint, electron_tolerance);
    return checkpoint;
}

} // namespace fde
