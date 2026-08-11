#include "fde_session_contract.h"

#include "source_lcao/module_fde/runtime/fde_runtime_config.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace fde
{

namespace
{

void require(const bool condition, const std::string& field)
{
    if (!condition)
    {
        throw std::invalid_argument(
            "FDE session request changes initialized field " + field);
    }
}

bool equal_fragments(const std::vector<FragmentDefinition>& left,
                     const std::vector<FragmentDefinition>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].label != right[index].label
            || left[index].neutral_valence_electrons
                   != right[index].neutral_valence_electrons
            || left[index].atom_indices != right[index].atom_indices)
        {
            return false;
        }
    }
    return true;
}

bool equal_assignments(const std::vector<FragmentChargeSpin>& left,
                       const std::vector<FragmentChargeSpin>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].fragment_label != right[index].fragment_label
            || left[index].charge != right[index].charge
            || left[index].spin_projection != right[index].spin_projection)
        {
            return false;
        }
    }
    return true;
}

bool equal_states(const std::vector<RuntimeStateDefinition>& left,
                  const std::vector<RuntimeStateDefinition>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].state.label != right[index].state.label
            || left[index].total_charge != right[index].total_charge
            || left[index].total_spin_projection
                   != right[index].total_spin_projection
            || !equal_assignments(left[index].state.fragments,
                                  right[index].state.fragments))
        {
            return false;
        }
    }
    return true;
}

bool equal_artifacts(const std::vector<RuntimeArtifactPath>& left,
                     const std::vector<RuntimeArtifactPath>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].label != right[index].label
            || left[index].path != right[index].path)
        {
            return false;
        }
    }
    return true;
}

bool equal_frozen_labels(const std::vector<RuntimeArtifactPath>& left,
                         const std::vector<RuntimeArtifactPath>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].label != right[index].label)
        {
            return false;
        }
    }
    return true;
}

bool equal_diagonal_energies(const std::vector<RuntimeDiagonalEnergy>& left,
                             const std::vector<RuntimeDiagonalEnergy>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].state_label != right[index].state_label
            || left[index].energy_ry != right[index].energy_ry)
        {
            return false;
        }
    }
    return true;
}

} // namespace

void FdeSessionContract::validate_compatible(
    const FdeRuntimeConfig& initialized,
    const FdeRuntimeConfig& requested)
{
    // ACTIVE_DENSITY, FROZEN_DENSITY paths, and OUTPUT_PREFIX are the only
    // request fields that may change. Everything that determines allocated AO
    // or PW state remains fixed for the lifetime of one worker.
    require(initialized.schema_version == requested.schema_version,
            "schema_version");
    require(initialized.atom_count == requested.atom_count, "atom_count");
    require(equal_fragments(initialized.fragments, requested.fragments),
            "fragments");
    require(equal_states(initialized.states, requested.states), "states");
    require(initialized.active_state == requested.active_state,
            "active_state");
    require(initialized.active_fragment == requested.active_fragment,
            "active_fragment");
    require(equal_frozen_labels(initialized.frozen_density_artifacts,
                                requested.frozen_density_artifacts),
            "frozen_density_labels");
    require(equal_artifacts(initialized.determinant_artifacts,
                            requested.determinant_artifacts),
            "determinant_artifacts");
    require(equal_artifacts(initialized.linearized_state_artifacts,
                            requested.linearized_state_artifacts),
            "linearized_state_artifacts");
    require(equal_diagonal_energies(initialized.diagonal_energies,
                                    requested.diagonal_energies),
            "diagonal_energies");
    require(initialized.ao_overlap_path == requested.ao_overlap_path,
            "ao_overlap_path");
    require(initialized.fragment_xc == requested.fragment_xc,
            "fragment_xc");
    require(initialized.embedding_xc == requested.embedding_xc,
            "embedding_xc");
    require(initialized.coupling_provider == requested.coupling_provider,
            "coupling_provider");
    require(initialized.transition_density_trace_tolerance
                == requested.transition_density_trace_tolerance,
            "transition_density_trace_tolerance");
    require(initialized.kinetic_functional == requested.kinetic_functional,
            "kinetic_functional");
    require(initialized.density_floor_bohr3 == requested.density_floor_bohr3,
            "density_floor_bohr3");
    require(initialized.maximum_scf_iterations
                == requested.maximum_scf_iterations,
            "maximum_scf_iterations");
    require(initialized.scf_density_tolerance
                == requested.scf_density_tolerance,
            "scf_density_tolerance");
    require(initialized.electron_tolerance == requested.electron_tolerance,
            "electron_tolerance");
    require(initialized.mixing_beta == requested.mixing_beta, "mixing_beta");
    require(initialized.maximum_freeze_thaw_cycles
                == requested.maximum_freeze_thaw_cycles,
            "maximum_freeze_thaw_cycles");
    require(initialized.freeze_thaw_density_tolerance
                == requested.freeze_thaw_density_tolerance,
            "freeze_thaw_density_tolerance");
    require(initialized.energy_tolerance_ry == requested.energy_tolerance_ry,
            "energy_tolerance_ry");
    require(initialized.update_order == requested.update_order, "update_order");
    require(initialized.k_state_labels == requested.k_state_labels,
            "k_state_labels");
    require(initialized.l_fragment_labels == requested.l_fragment_labels,
            "l_fragment_labels");
    require(initialized.m_fragment_labels == requested.m_fragment_labels,
            "m_fragment_labels");
    require(initialized.singular_value_tolerance
                == requested.singular_value_tolerance,
            "singular_value_tolerance");
    require(initialized.overlap_eigenvalue_cutoff
                == requested.overlap_eigenvalue_cutoff,
            "overlap_eigenvalue_cutoff");
    require(initialized.symmetry_tolerance == requested.symmetry_tolerance,
            "symmetry_tolerance");
    require(initialized.residual_tolerance == requested.residual_tolerance,
            "residual_tolerance");
    require(initialized.minimum_root_overlap == requested.minimum_root_overlap,
            "minimum_root_overlap");
    require(initialized.calculate_force == requested.calculate_force,
            "calculate_force");
    require(initialized.hartree_reciprocity_tolerance_ry
                == requested.hartree_reciprocity_tolerance_ry,
            "hartree_reciprocity_tolerance_ry");
}

} // namespace fde
