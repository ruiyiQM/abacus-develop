#include "fde_multi_fragment_freeze_thaw.h"

#include <cmath>
#include <istream>
#include <limits>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fde
{

namespace
{

const int checkpoint_schema_version = 1;

void validate_energy_labels(const MultiFragmentEnergyLedger& energy,
                            const std::vector<FrozenDensityArtifact>& fragments)
{
    MultiFragmentCanonicalEnergy::validate(energy);
    if (energy.fragments.size() != fragments.size())
    {
        throw std::invalid_argument("FDE multi-fragment energy has the wrong fragment count");
    }
    std::set<std::string> artifact_labels;
    std::set<std::string> energy_labels;
    for (std::size_t index = 0; index < fragments.size(); ++index)
    {
        artifact_labels.insert(fragments[index].fragment_label);
        energy_labels.insert(energy.fragments[index].fragment_label);
    }
    if (artifact_labels != energy_labels)
    {
        throw std::invalid_argument("FDE multi-fragment energy labels do not match artifacts");
    }
}

void validate_checkpoint(const MultiFragmentFreezeThawCheckpoint& checkpoint,
                         const double electron_tolerance)
{
    if (checkpoint.schema_version != checkpoint_schema_version)
    {
        throw std::invalid_argument("Unsupported FDE multi-fragment checkpoint version");
    }
    DensityArtifactIO::validate_compatible_set(checkpoint.fragments, electron_tolerance);
    if (checkpoint.completed_cycles < 0)
    {
        throw std::invalid_argument("FDE multi-fragment checkpoint cycle is negative");
    }
    for (std::size_t index = 0; index < checkpoint.fragments.size(); ++index)
    {
        if (!checkpoint.fragments[index].scf_converged
            || checkpoint.fragments[index].freeze_thaw_cycle != checkpoint.completed_cycles)
        {
            throw std::invalid_argument(
                "FDE multi-fragment checkpoint contains inconsistent fragment cycles");
        }
    }
    if (!std::isfinite(checkpoint.density_residual) || checkpoint.density_residual < 0.0
        || !std::isfinite(checkpoint.energy_change_ry) || checkpoint.energy_change_ry < 0.0)
    {
        throw std::invalid_argument("FDE multi-fragment checkpoint residuals are invalid");
    }
    validate_energy_labels(checkpoint.energy, checkpoint.fragments);
}

void validate_controls(const MultiFragmentFreezeThawControls& controls,
                       const std::size_t fragment_count)
{
    if (controls.maximum_cycles <= 0
        || !std::isfinite(controls.density_tolerance) || controls.density_tolerance <= 0.0
        || !std::isfinite(controls.energy_tolerance_ry) || controls.energy_tolerance_ry <= 0.0
        || !std::isfinite(controls.electron_tolerance) || controls.electron_tolerance < 0.0
        || controls.update_order.size() != fragment_count)
    {
        throw std::invalid_argument("FDE multi-fragment freeze-thaw controls are invalid");
    }
    std::set<std::size_t> order;
    for (std::size_t index = 0; index < controls.update_order.size(); ++index)
    {
        if (controls.update_order[index] >= fragment_count
            || !order.insert(controls.update_order[index]).second)
        {
            throw std::invalid_argument(
                "FDE multi-fragment update order must be a fragment-index permutation");
        }
    }
}

std::vector<FrozenDensityArtifact> environment_without(
    const std::vector<FrozenDensityArtifact>& fragments,
    const std::size_t active_index)
{
    std::vector<FrozenDensityArtifact> environment;
    environment.reserve(fragments.size() - 1);
    for (std::size_t index = 0; index < fragments.size(); ++index)
    {
        if (index != active_index)
        {
            environment.push_back(fragments[index]);
        }
    }
    return environment;
}

FrozenDensityArtifact update_fragment(
    const std::vector<FrozenDensityArtifact>& fragments,
    const std::size_t active_index,
    const MultiFragmentFreezeThawStepRunner& runner,
    const double electron_tolerance)
{
    const FrozenDensityArtifact& active = fragments[active_index];
    const std::vector<FrozenDensityArtifact> environment
        = environment_without(fragments, active_index);
    const FrozenDensityArtifact updated = runner.update(active, environment);
    std::vector<FrozenDensityArtifact> compatible = environment;
    compatible.push_back(updated);
    DensityArtifactIO::validate_compatible_set(compatible, electron_tolerance);
    if (updated.fragment_label != active.fragment_label
        || updated.state_label != active.state_label
        || updated.alpha_electrons != active.alpha_electrons
        || updated.beta_electrons != active.beta_electrons)
    {
        throw std::invalid_argument(
            "FDE multi-fragment update changed subsystem identity or populations");
    }
    if (!updated.scf_converged)
    {
        throw std::runtime_error("FDE multi-fragment one-way SCF did not converge");
    }
    if (updated.freeze_thaw_cycle != active.freeze_thaw_cycle + 1)
    {
        throw std::invalid_argument("FDE multi-fragment update returned the wrong cycle");
    }
    return updated;
}

double density_residual(const std::vector<FrozenDensityArtifact>& before,
                        const std::vector<FrozenDensityArtifact>& after)
{
    const std::size_t grid_size = before[0].rho_alpha_bohr3.size();
    const double volume_element
        = before[0].cell_volume_bohr3 / static_cast<double>(grid_size);
    double squared = 0.0;
    for (std::size_t fragment = 0; fragment < before.size(); ++fragment)
    {
        for (std::size_t index = 0; index < grid_size; ++index)
        {
            const double alpha = after[fragment].rho_alpha_bohr3[index]
                                 - before[fragment].rho_alpha_bohr3[index];
            const double beta = after[fragment].rho_beta_bohr3[index]
                                - before[fragment].rho_beta_bohr3[index];
            squared += (alpha * alpha + beta * beta) * volume_element;
        }
    }
    return std::sqrt(squared);
}

void require_token(std::istream& input, const char* expected)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        throw std::invalid_argument(std::string("FDE multi-fragment checkpoint expected token ")
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
        throw std::invalid_argument("FDE multi-fragment checkpoint block size is invalid");
    }
    char separator = '\0';
    input.get(separator);
    if (separator != '\n')
    {
        throw std::invalid_argument("FDE multi-fragment checkpoint block header is malformed");
    }
    std::string content(size, '\0');
    if (size > 0)
    {
        input.read(&content[0], static_cast<std::streamsize>(size));
    }
    input.get(separator);
    if (!input || separator != '\n')
    {
        throw std::invalid_argument("FDE multi-fragment checkpoint block is truncated");
    }
    return content;
}

} // namespace

MultiFragmentFreezeThawCheckpoint MultiFragmentFreezeThawWorkflow::initialize(
    const std::vector<FrozenDensityArtifact>& fragments,
    const MultiFragmentFreezeThawStepRunner& runner,
    const double electron_tolerance)
{
    DensityArtifactIO::validate_compatible_set(fragments, electron_tolerance);
    const int cycle = fragments[0].freeze_thaw_cycle;
    for (std::size_t index = 0; index < fragments.size(); ++index)
    {
        if (!fragments[index].scf_converged || fragments[index].freeze_thaw_cycle != cycle)
        {
            throw std::invalid_argument(
                "FDE multi-fragment initialization requires matching converged cycles");
        }
    }
    MultiFragmentFreezeThawCheckpoint checkpoint;
    checkpoint.schema_version = checkpoint_schema_version;
    checkpoint.fragments = fragments;
    checkpoint.energy = runner.canonical_energy(fragments);
    checkpoint.completed_cycles = cycle;
    checkpoint.converged = false;
    checkpoint.density_residual = 0.0;
    checkpoint.energy_change_ry = 0.0;
    validate_checkpoint(checkpoint, electron_tolerance);
    return checkpoint;
}

MultiFragmentFreezeThawCheckpoint MultiFragmentFreezeThawWorkflow::run(
    const MultiFragmentFreezeThawCheckpoint& checkpoint,
    const MultiFragmentFreezeThawControls& controls,
    const MultiFragmentFreezeThawStepRunner& runner)
{
    validate_controls(controls, checkpoint.fragments.size());
    validate_checkpoint(checkpoint, controls.electron_tolerance);
    if (checkpoint.converged)
    {
        return checkpoint;
    }
    if (checkpoint.completed_cycles >= controls.maximum_cycles)
    {
        throw std::invalid_argument(
            "FDE multi-fragment maximum cycle is not beyond the checkpoint");
    }

    MultiFragmentFreezeThawCheckpoint result = checkpoint;
    while (result.completed_cycles < controls.maximum_cycles && !result.converged)
    {
        const std::vector<FrozenDensityArtifact> before = result.fragments;
        const double old_energy = MultiFragmentCanonicalEnergy::total(result.energy);
        for (std::size_t order_index = 0; order_index < controls.update_order.size(); ++order_index)
        {
            const std::size_t active_index = controls.update_order[order_index];
            result.fragments[active_index]
                = update_fragment(result.fragments,
                                  active_index,
                                  runner,
                                  controls.electron_tolerance);
        }
        ++result.completed_cycles;
        result.density_residual = density_residual(before, result.fragments);
        result.energy = runner.canonical_energy(result.fragments);
        validate_energy_labels(result.energy, result.fragments);
        result.energy_change_ry
            = std::fabs(MultiFragmentCanonicalEnergy::total(result.energy) - old_energy);
        result.converged = result.density_residual <= controls.density_tolerance
                           && result.energy_change_ry <= controls.energy_tolerance_ry;
    }
    validate_checkpoint(result, controls.electron_tolerance);
    return result;
}

void MultiFragmentFreezeThawWorkflow::write_checkpoint(
    std::ostream& output,
    const MultiFragmentFreezeThawCheckpoint& checkpoint)
{
    validate_checkpoint(checkpoint, 1.0e-8);
    output.precision(17);
    output << "FDE_MULTI_FRAGMENT_FREEZE_THAW_CHECKPOINT " << checkpoint.schema_version << '\n';
    output << "STATUS " << checkpoint.completed_cycles << ' '
           << (checkpoint.converged ? 1 : 0) << ' '
           << checkpoint.density_residual << ' '
           << checkpoint.energy_change_ry << '\n';
    output << "FRAGMENTS " << checkpoint.fragments.size() << '\n';
    for (std::size_t index = 0; index < checkpoint.fragments.size(); ++index)
    {
        std::ostringstream artifact_output;
        DensityArtifactIO::write(artifact_output, checkpoint.fragments[index]);
        write_block(output, "FRAGMENT_ARTIFACT_BYTES", artifact_output.str());
    }
    std::ostringstream energy_output;
    MultiFragmentCanonicalEnergy::write(energy_output, checkpoint.energy);
    write_block(output, "ENERGY_LEDGER_BYTES", energy_output.str());
    output << "END\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE multi-fragment checkpoint");
    }
}

MultiFragmentFreezeThawCheckpoint MultiFragmentFreezeThawWorkflow::read_checkpoint(
    std::istream& input,
    const double electron_tolerance)
{
    MultiFragmentFreezeThawCheckpoint checkpoint;
    require_token(input, "FDE_MULTI_FRAGMENT_FREEZE_THAW_CHECKPOINT");
    input >> checkpoint.schema_version;
    require_token(input, "STATUS");
    int converged = 0;
    input >> checkpoint.completed_cycles >> converged
          >> checkpoint.density_residual >> checkpoint.energy_change_ry;
    if (converged != 0 && converged != 1)
    {
        throw std::invalid_argument("FDE multi-fragment convergence flag is invalid");
    }
    checkpoint.converged = converged == 1;
    require_token(input, "FRAGMENTS");
    std::size_t fragment_count = 0;
    input >> fragment_count;
    checkpoint.fragments.resize(fragment_count);
    for (std::size_t index = 0; index < fragment_count; ++index)
    {
        const std::string content = read_block(input, "FRAGMENT_ARTIFACT_BYTES");
        std::istringstream artifact_input(content);
        checkpoint.fragments[index]
            = DensityArtifactIO::read(artifact_input, electron_tolerance);
    }
    const std::string energy_content = read_block(input, "ENERGY_LEDGER_BYTES");
    std::istringstream energy_input(energy_content);
    checkpoint.energy = MultiFragmentCanonicalEnergy::read(energy_input);
    require_token(input, "END");
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE multi-fragment checkpoint has trailing content");
    }
    validate_checkpoint(checkpoint, electron_tolerance);
    return checkpoint;
}

} // namespace fde
