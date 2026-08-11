#include "source_lcao/module_fde/runtime/fde_runtime_config.h"

#include "source_lcao/module_fde/embedding/fde_xc_policy.h"
#include "source_lcao/module_fde/coupling/fde_coupling_policy.h"

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fde
{

namespace
{

std::runtime_error parse_error(const std::size_t line, const std::string& message)
{
    std::ostringstream output;
    output << "FDE_CONFIG line " << line << ": " << message;
    return std::runtime_error(output.str());
}

void require_end(std::istringstream& line, const std::size_t line_number)
{
    std::string extra;
    if (line >> extra)
    {
        throw parse_error(line_number, "unexpected trailing token " + extra);
    }
}

template <typename T>
T read_value(std::istringstream& line,
             const std::size_t line_number,
             const std::string& name)
{
    T value;
    if (!(line >> value))
    {
        throw parse_error(line_number, "missing or invalid " + name);
    }
    return value;
}

bool read_bool(std::istringstream& line, const std::size_t line_number)
{
    const std::string value = read_value<std::string>(line, line_number, "Boolean value");
    if (value == "true" || value == "1")
    {
        return true;
    }
    if (value == "false" || value == "0")
    {
        return false;
    }
    throw parse_error(line_number, "Boolean value must be true, false, 1, or 0");
}

std::vector<std::string> read_labels(std::istringstream& line,
                                     const std::size_t line_number)
{
    const std::size_t count = read_value<std::size_t>(line, line_number, "label count");
    std::vector<std::string> labels(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        labels[index] = read_value<std::string>(line, line_number, "label");
    }
    return labels;
}

bool finite_positive(const double value)
{
    return std::isfinite(value) && value > 0.0;
}

bool contains(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

void require_unique(const std::vector<std::string>& labels, const std::string& kind)
{
    std::vector<std::string> sorted = labels;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("FDE_CONFIG contains a duplicate " + kind + " label");
    }
}

} // namespace

FdeRuntimeConfig::FdeRuntimeConfig()
    : schema_version(1),
      atom_count(0),
      fragment_xc("pbe"),
      embedding_xc("pbe"),
      kinetic_functional(KineticFunctional::Pw91k),
      density_floor_bohr3(1.0e-12),
      maximum_scf_iterations(100),
      scf_density_tolerance(1.0e-8),
      electron_tolerance(1.0e-8),
      mixing_beta(0.3),
      maximum_freeze_thaw_cycles(20),
      freeze_thaw_density_tolerance(1.0e-7),
      energy_tolerance_ry(1.0e-8),
      coupling_provider("symmetric_linearized"),
      transition_density_trace_tolerance(1.0e-8),
      singular_value_tolerance(1.0e-10),
      overlap_eigenvalue_cutoff(1.0e-9),
      symmetry_tolerance(1.0e-10),
      residual_tolerance(1.0e-8),
      minimum_root_overlap(0.5),
      calculate_force(false),
      hartree_reciprocity_tolerance_ry(1.0e-8)
{
}

std::size_t FdeRuntimeConfigIO::state_index(const FdeRuntimeConfig& config,
                                            const std::string& label)
{
    for (std::size_t index = 0; index < config.states.size(); ++index)
    {
        if (config.states[index].state.label == label)
        {
            return index;
        }
    }
    throw std::invalid_argument("FDE_CONFIG references unknown state " + label);
}

std::size_t FdeRuntimeConfigIO::fragment_index(const FdeRuntimeConfig& config,
                                               const std::string& label)
{
    for (std::size_t index = 0; index < config.fragments.size(); ++index)
    {
        if (config.fragments[index].label == label)
        {
            return index;
        }
    }
    throw std::invalid_argument("FDE_CONFIG references unknown fragment " + label);
}

void FdeRuntimeConfigIO::validate(const FdeRuntimeConfig& config)
{
    if (config.schema_version != 1)
    {
        throw std::invalid_argument("FDE_CONFIG schema version must be 1");
    }
    if (config.atom_count == 0 || config.fragments.size() < 2 || config.states.size() < 2)
    {
        throw std::invalid_argument("FDE_CONFIG requires atoms, at least two fragments, and at least two states");
    }

    std::vector<std::string> fragment_labels;
    for (std::size_t index = 0; index < config.fragments.size(); ++index)
    {
        fragment_labels.push_back(config.fragments[index].label);
    }
    require_unique(fragment_labels, "fragment");
    StateDefinition::validate_partition(config.fragments, config.atom_count);

    std::vector<std::string> state_labels;
    for (std::size_t index = 0; index < config.states.size(); ++index)
    {
        state_labels.push_back(config.states[index].state.label);
        StateDefinition::validate_state(config.fragments,
                                        config.states[index].state,
                                        config.states[index].total_charge,
                                        config.states[index].total_spin_projection);
    }
    require_unique(state_labels, "state");

    if (!config.active_state.empty())
    {
        FdeRuntimeConfigIO::state_index(config, config.active_state);
    }
    if (!config.active_fragment.empty())
    {
        FdeRuntimeConfigIO::fragment_index(config, config.active_fragment);
    }
    if (config.active_state.empty() != config.active_fragment.empty())
    {
        throw std::invalid_argument("FDE_CONFIG ACTIVE_STATE and ACTIVE_FRAGMENT must be specified together");
    }

    for (std::size_t index = 0; index < config.frozen_density_artifacts.size(); ++index)
    {
        FdeRuntimeConfigIO::fragment_index(config, config.frozen_density_artifacts[index].label);
        if (config.frozen_density_artifacts[index].path.empty())
        {
            throw std::invalid_argument("FDE_CONFIG frozen-density path must not be empty");
        }
    }
    for (std::size_t index = 0; index < config.determinant_artifacts.size(); ++index)
    {
        FdeRuntimeConfigIO::state_index(config, config.determinant_artifacts[index].label);
        if (config.determinant_artifacts[index].path.empty())
        {
            throw std::invalid_argument("FDE_CONFIG determinant path must not be empty");
        }
    }
    for (std::size_t index = 0; index < config.linearized_state_artifacts.size(); ++index)
    {
        FdeRuntimeConfigIO::state_index(config,
                                        config.linearized_state_artifacts[index].label);
        if (config.linearized_state_artifacts[index].path.empty())
        {
            throw std::invalid_argument("FDE_CONFIG linearized-state path must not be empty");
        }
    }
    for (std::size_t index = 0; index < config.diagonal_energies.size(); ++index)
    {
        FdeRuntimeConfigIO::state_index(config, config.diagonal_energies[index].state_label);
        if (!std::isfinite(config.diagonal_energies[index].energy_ry))
        {
            throw std::invalid_argument("FDE_CONFIG diagonal energy must be finite");
        }
    }

    if (config.output_prefix.empty())
    {
        throw std::invalid_argument("FDE_CONFIG OUTPUT_PREFIX must not be empty");
    }
    FdeXcPolicy::canonical_fragment(config.fragment_xc);
    FdeXcPolicy::canonical_embedding(config.embedding_xc);
    FdeCouplingPolicy::canonical_provider(config.coupling_provider);
    if (!finite_positive(config.density_floor_bohr3)
        || config.maximum_scf_iterations < 1
        || !finite_positive(config.scf_density_tolerance)
        || !finite_positive(config.electron_tolerance)
        || !std::isfinite(config.mixing_beta) || config.mixing_beta <= 0.0
        || config.mixing_beta > 1.0 || config.maximum_freeze_thaw_cycles < 1
        || !finite_positive(config.freeze_thaw_density_tolerance)
        || !finite_positive(config.energy_tolerance_ry)
        || !finite_positive(config.transition_density_trace_tolerance)
        || !finite_positive(config.singular_value_tolerance)
        || !finite_positive(config.overlap_eigenvalue_cutoff)
        || !finite_positive(config.symmetry_tolerance)
        || !finite_positive(config.residual_tolerance)
        || !std::isfinite(config.minimum_root_overlap)
        || config.minimum_root_overlap < 0.0 || config.minimum_root_overlap > 1.0
        || !finite_positive(config.hartree_reciprocity_tolerance_ry))
    {
        throw std::invalid_argument("FDE_CONFIG contains invalid numerical controls");
    }

    if (!config.update_order.empty())
    {
        require_unique(config.update_order, "update-order fragment");
        if (config.update_order.size() != config.fragments.size())
        {
            throw std::invalid_argument("FDE_CONFIG UPDATE_ORDER must contain every fragment once");
        }
        for (std::size_t index = 0; index < config.update_order.size(); ++index)
        {
            FdeRuntimeConfigIO::fragment_index(config, config.update_order[index]);
        }
    }
    if (!config.k_state_labels.empty() && config.k_state_labels.size() < 2)
    {
        throw std::invalid_argument("FDE_CONFIG K_STATES must select at least two states");
    }
    require_unique(config.k_state_labels, "K-state");
    for (std::size_t index = 0; index < config.k_state_labels.size(); ++index)
    {
        FdeRuntimeConfigIO::state_index(config, config.k_state_labels[index]);
    }
    require_unique(config.l_fragment_labels, "L-fragment");
    require_unique(config.m_fragment_labels, "M-fragment");
    for (std::size_t index = 0; index < config.l_fragment_labels.size(); ++index)
    {
        FdeRuntimeConfigIO::fragment_index(config, config.l_fragment_labels[index]);
        if (!contains(config.m_fragment_labels, config.l_fragment_labels[index]))
        {
            throw std::invalid_argument("FDE_CONFIG L_FRAGMENTS must be a subset of M_FRAGMENTS");
        }
    }
    for (std::size_t index = 0; index < config.m_fragment_labels.size(); ++index)
    {
        FdeRuntimeConfigIO::fragment_index(config, config.m_fragment_labels[index]);
    }
}

FdeRuntimeConfig FdeRuntimeConfigIO::read(std::istream& input)
{
    FdeRuntimeConfig config;
    bool header_seen = false;
    bool end_seen = false;
    std::string raw_line;
    std::size_t line_number = 0;
    while (std::getline(input, raw_line))
    {
        ++line_number;
        const std::size_t comment = raw_line.find('#');
        if (comment != std::string::npos)
        {
            raw_line.erase(comment);
        }
        std::istringstream line(raw_line);
        std::string key;
        if (!(line >> key))
        {
            continue;
        }
        if (!header_seen)
        {
            if (key != "FDE_CONFIG")
            {
                throw parse_error(line_number, "first record must be FDE_CONFIG 1");
            }
            config.schema_version = read_value<int>(line, line_number, "schema version");
            require_end(line, line_number);
            header_seen = true;
            continue;
        }
        if (end_seen)
        {
            throw parse_error(line_number, "records after END_FDE_CONFIG are not allowed");
        }
        if (key == "END_FDE_CONFIG")
        {
            require_end(line, line_number);
            end_seen = true;
            continue;
        }
        if (key == "ATOM_COUNT")
        {
            config.atom_count = read_value<std::size_t>(line, line_number, "atom count");
        }
        else if (key == "FRAGMENT")
        {
            FragmentDefinition fragment;
            fragment.label = read_value<std::string>(line, line_number, "fragment label");
            fragment.neutral_valence_electrons
                = read_value<int>(line, line_number, "neutral valence-electron count");
            const std::size_t count = read_value<std::size_t>(line, line_number, "fragment atom count");
            for (std::size_t index = 0; index < count; ++index)
            {
                fragment.atom_indices.push_back(
                    read_value<std::size_t>(line, line_number, "atom index"));
            }
            config.fragments.push_back(fragment);
        }
        else if (key == "STATE")
        {
            RuntimeStateDefinition runtime_state;
            runtime_state.state.label = read_value<std::string>(line, line_number, "state label");
            runtime_state.total_charge = read_value<int>(line, line_number, "total charge");
            runtime_state.total_spin_projection
                = read_value<int>(line, line_number, "total spin projection");
            const std::size_t count = read_value<std::size_t>(line, line_number, "assignment count");
            for (std::size_t index = 0; index < count; ++index)
            {
                FragmentChargeSpin assignment;
                assignment.fragment_label
                    = read_value<std::string>(line, line_number, "assigned fragment label");
                assignment.charge = read_value<int>(line, line_number, "fragment charge");
                assignment.spin_projection
                    = read_value<int>(line, line_number, "fragment spin projection");
                runtime_state.state.fragments.push_back(assignment);
            }
            config.states.push_back(runtime_state);
        }
        else if (key == "ACTIVE_STATE")
        {
            config.active_state = read_value<std::string>(line, line_number, "active state");
        }
        else if (key == "ACTIVE_FRAGMENT")
        {
            config.active_fragment = read_value<std::string>(line, line_number, "active fragment");
        }
        else if (key == "ACTIVE_DENSITY")
        {
            config.active_density_path
                = read_value<std::string>(line, line_number, "active-density path");
        }
        else if (key == "FROZEN_DENSITY" || key == "DETERMINANT"
                 || key == "LINEARIZED_STATE")
        {
            RuntimeArtifactPath artifact;
            artifact.label = read_value<std::string>(line, line_number, "artifact label");
            artifact.path = read_value<std::string>(line, line_number, "artifact path");
            if (key == "FROZEN_DENSITY")
            {
                config.frozen_density_artifacts.push_back(artifact);
            }
            else if (key == "DETERMINANT")
            {
                config.determinant_artifacts.push_back(artifact);
            }
            else
            {
                config.linearized_state_artifacts.push_back(artifact);
            }
        }
        else if (key == "DIAGONAL_ENERGY_RY")
        {
            RuntimeDiagonalEnergy energy;
            energy.state_label = read_value<std::string>(line, line_number, "state label");
            energy.energy_ry = read_value<double>(line, line_number, "diagonal energy");
            config.diagonal_energies.push_back(energy);
        }
        else if (key == "AO_OVERLAP")
        {
            config.ao_overlap_path = read_value<std::string>(line, line_number, "AO-overlap path");
        }
        else if (key == "OUTPUT_PREFIX")
        {
            config.output_prefix = read_value<std::string>(line, line_number, "output prefix");
        }
        else if (key == "KEDF")
        {
            const std::string value = read_value<std::string>(line, line_number, "KEDF");
            if (value == "pw91k" || value == "lc94")
            {
                config.kinetic_functional = KineticFunctional::Pw91k;
            }
            else if (value == "thomas_fermi" || value == "tf")
            {
                config.kinetic_functional = KineticFunctional::ThomasFermi;
            }
            else if (value == "revapbek")
            {
                config.kinetic_functional = KineticFunctional::RevApbek;
            }
            else
            {
                throw parse_error(
                    line_number,
                    "KEDF must be pw91k, lc94, thomas_fermi, tf, or revapbek");
            }
        }
        else if (key == "FRAGMENT_XC")
        {
            config.fragment_xc = FdeXcPolicy::canonical_fragment(
                read_value<std::string>(line, line_number, "fragment XC"));
        }
        else if (key == "EMBEDDING_XC")
        {
            config.embedding_xc = FdeXcPolicy::canonical_embedding(
                read_value<std::string>(line, line_number, "embedding XC"));
        }
        else if (key == "COUPLING_PROVIDER")
        {
            config.coupling_provider = FdeCouplingPolicy::canonical_provider(
                read_value<std::string>(line, line_number, "coupling provider"));
        }
        else if (key == "TRANSITION_DENSITY_TRACE_TOLERANCE")
        {
            config.transition_density_trace_tolerance
                = read_value<double>(line,
                                     line_number,
                                     "transition-density trace tolerance");
        }
        else if (key == "DENSITY_FLOOR_BOHR3")
        {
            config.density_floor_bohr3 = read_value<double>(line, line_number, "density floor");
        }
        else if (key == "MAX_SCF_ITERATIONS")
        {
            config.maximum_scf_iterations = read_value<int>(line, line_number, "SCF iteration limit");
        }
        else if (key == "SCF_DENSITY_TOLERANCE")
        {
            config.scf_density_tolerance = read_value<double>(line, line_number, "SCF density tolerance");
        }
        else if (key == "ELECTRON_TOLERANCE")
        {
            config.electron_tolerance = read_value<double>(line, line_number, "electron tolerance");
        }
        else if (key == "MIXING_BETA")
        {
            config.mixing_beta = read_value<double>(line, line_number, "mixing beta");
        }
        else if (key == "MAX_FREEZE_THAW_CYCLES")
        {
            config.maximum_freeze_thaw_cycles = read_value<int>(line, line_number, "freeze-thaw cycle limit");
        }
        else if (key == "FREEZE_THAW_DENSITY_TOLERANCE")
        {
            config.freeze_thaw_density_tolerance
                = read_value<double>(line, line_number, "freeze-thaw density tolerance");
        }
        else if (key == "ENERGY_TOLERANCE_RY")
        {
            config.energy_tolerance_ry = read_value<double>(line, line_number, "energy tolerance");
        }
        else if (key == "UPDATE_ORDER")
        {
            config.update_order = read_labels(line, line_number);
        }
        else if (key == "K_STATES")
        {
            config.k_state_labels = read_labels(line, line_number);
        }
        else if (key == "L_FRAGMENTS")
        {
            config.l_fragment_labels = read_labels(line, line_number);
        }
        else if (key == "M_FRAGMENTS")
        {
            config.m_fragment_labels = read_labels(line, line_number);
        }
        else if (key == "SINGULAR_VALUE_TOLERANCE")
        {
            config.singular_value_tolerance = read_value<double>(line, line_number, "singular-value tolerance");
        }
        else if (key == "OVERLAP_EIGENVALUE_CUTOFF")
        {
            config.overlap_eigenvalue_cutoff = read_value<double>(line, line_number, "overlap cutoff");
        }
        else if (key == "SYMMETRY_TOLERANCE")
        {
            config.symmetry_tolerance = read_value<double>(line, line_number, "symmetry tolerance");
        }
        else if (key == "RESIDUAL_TOLERANCE")
        {
            config.residual_tolerance = read_value<double>(line, line_number, "residual tolerance");
        }
        else if (key == "MINIMUM_ROOT_OVERLAP")
        {
            config.minimum_root_overlap = read_value<double>(line, line_number, "minimum root overlap");
        }
        else if (key == "CALCULATE_FORCE")
        {
            config.calculate_force = read_bool(line, line_number);
        }
        else if (key == "HARTREE_RECIPROCITY_TOLERANCE_RY")
        {
            config.hartree_reciprocity_tolerance_ry
                = read_value<double>(line, line_number, "Hartree reciprocity tolerance");
        }
        else
        {
            throw parse_error(line_number, "unknown keyword " + key);
        }
        require_end(line, line_number);
    }
    if (!header_seen || !end_seen)
    {
        throw std::runtime_error("FDE_CONFIG requires FDE_CONFIG 1 and END_FDE_CONFIG records");
    }
    FdeRuntimeConfigIO::validate(config);
    return config;
}

void FdeRuntimeConfigIO::write(std::ostream& output, const FdeRuntimeConfig& config)
{
    FdeRuntimeConfigIO::validate(config);
    output << "FDE_CONFIG " << config.schema_version << '\n';
    output << "ATOM_COUNT " << config.atom_count << '\n';
    for (std::size_t index = 0; index < config.fragments.size(); ++index)
    {
        const FragmentDefinition& fragment = config.fragments[index];
        output << "FRAGMENT " << fragment.label << ' ' << fragment.neutral_valence_electrons
               << ' ' << fragment.atom_indices.size();
        for (std::size_t atom = 0; atom < fragment.atom_indices.size(); ++atom)
        {
            output << ' ' << fragment.atom_indices[atom];
        }
        output << '\n';
    }
    for (std::size_t index = 0; index < config.states.size(); ++index)
    {
        const RuntimeStateDefinition& state = config.states[index];
        output << "STATE " << state.state.label << ' ' << state.total_charge << ' '
               << state.total_spin_projection << ' ' << state.state.fragments.size();
        for (std::size_t fragment = 0; fragment < state.state.fragments.size(); ++fragment)
        {
            const FragmentChargeSpin& assignment = state.state.fragments[fragment];
            output << ' ' << assignment.fragment_label << ' ' << assignment.charge << ' '
                   << assignment.spin_projection;
        }
        output << '\n';
    }
    if (!config.active_state.empty())
    {
        output << "ACTIVE_STATE " << config.active_state << '\n';
        output << "ACTIVE_FRAGMENT " << config.active_fragment << '\n';
    }
    if (!config.active_density_path.empty())
    {
        output << "ACTIVE_DENSITY " << config.active_density_path << '\n';
    }
    for (std::size_t index = 0; index < config.frozen_density_artifacts.size(); ++index)
    {
        output << "FROZEN_DENSITY " << config.frozen_density_artifacts[index].label << ' '
               << config.frozen_density_artifacts[index].path << '\n';
    }
    for (std::size_t index = 0; index < config.determinant_artifacts.size(); ++index)
    {
        output << "DETERMINANT " << config.determinant_artifacts[index].label << ' '
               << config.determinant_artifacts[index].path << '\n';
    }
    for (std::size_t index = 0; index < config.linearized_state_artifacts.size(); ++index)
    {
        output << "LINEARIZED_STATE " << config.linearized_state_artifacts[index].label << ' '
               << config.linearized_state_artifacts[index].path << '\n';
    }
    for (std::size_t index = 0; index < config.diagonal_energies.size(); ++index)
    {
        output << "DIAGONAL_ENERGY_RY " << config.diagonal_energies[index].state_label << ' '
               << config.diagonal_energies[index].energy_ry << '\n';
    }
    if (!config.ao_overlap_path.empty())
    {
        output << "AO_OVERLAP " << config.ao_overlap_path << '\n';
    }
    output << "OUTPUT_PREFIX " << config.output_prefix << '\n';
    output << "FRAGMENT_XC " << FdeXcPolicy::canonical_fragment(config.fragment_xc)
           << '\n';
    output << "EMBEDDING_XC " << FdeXcPolicy::canonical_embedding(config.embedding_xc)
           << '\n';
    output << "COUPLING_PROVIDER "
           << FdeCouplingPolicy::canonical_provider(config.coupling_provider)
           << '\n';
    output << "TRANSITION_DENSITY_TRACE_TOLERANCE "
           << config.transition_density_trace_tolerance << '\n';
    output << "KEDF " << kinetic_functional_name(config.kinetic_functional) << '\n';
    output << "DENSITY_FLOOR_BOHR3 " << config.density_floor_bohr3 << '\n';
    output << "MAX_SCF_ITERATIONS " << config.maximum_scf_iterations << '\n';
    output << "SCF_DENSITY_TOLERANCE " << config.scf_density_tolerance << '\n';
    output << "ELECTRON_TOLERANCE " << config.electron_tolerance << '\n';
    output << "MIXING_BETA " << config.mixing_beta << '\n';
    output << "MAX_FREEZE_THAW_CYCLES " << config.maximum_freeze_thaw_cycles << '\n';
    output << "FREEZE_THAW_DENSITY_TOLERANCE " << config.freeze_thaw_density_tolerance << '\n';
    output << "ENERGY_TOLERANCE_RY " << config.energy_tolerance_ry << '\n';
    const std::vector<std::pair<std::string, const std::vector<std::string>*> > selections = {
        std::make_pair("UPDATE_ORDER", &config.update_order),
        std::make_pair("K_STATES", &config.k_state_labels),
        std::make_pair("L_FRAGMENTS", &config.l_fragment_labels),
        std::make_pair("M_FRAGMENTS", &config.m_fragment_labels)};
    for (std::size_t selection = 0; selection < selections.size(); ++selection)
    {
        output << selections[selection].first << ' ' << selections[selection].second->size();
        for (std::size_t index = 0; index < selections[selection].second->size(); ++index)
        {
            output << ' ' << (*selections[selection].second)[index];
        }
        output << '\n';
    }
    output << "SINGULAR_VALUE_TOLERANCE " << config.singular_value_tolerance << '\n';
    output << "OVERLAP_EIGENVALUE_CUTOFF " << config.overlap_eigenvalue_cutoff << '\n';
    output << "SYMMETRY_TOLERANCE " << config.symmetry_tolerance << '\n';
    output << "RESIDUAL_TOLERANCE " << config.residual_tolerance << '\n';
    output << "MINIMUM_ROOT_OVERLAP " << config.minimum_root_overlap << '\n';
    output << "CALCULATE_FORCE " << (config.calculate_force ? "true" : "false") << '\n';
    output << "HARTREE_RECIPROCITY_TOLERANCE_RY "
           << config.hartree_reciprocity_tolerance_ry << '\n';
    output << "END_FDE_CONFIG\n";
}

} // namespace fde
