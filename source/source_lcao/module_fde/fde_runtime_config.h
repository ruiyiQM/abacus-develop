#ifndef FDE_RUNTIME_CONFIG_H
#define FDE_RUNTIME_CONFIG_H

#include "fde_semilocal_functional.h"
#include "fde_state.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

struct RuntimeStateDefinition
{
    DiabaticStateSpec state;
    int total_charge;
    int total_spin_projection;
};

struct RuntimeArtifactPath
{
    std::string label;
    std::string path;
};

struct RuntimeDiagonalEnergy
{
    std::string state_label;
    double energy_ry;
};

/**
 * Versioned, task-local configuration used by the native FDE runtime.
 *
 * ABACUS INPUT owns only the runtime selector and this file path. Fragment and
 * state data stay here so a geometry scan can create independent, restartable
 * subsystem jobs without adding process-wide workflow state to Parameter.
 */
struct FdeRuntimeConfig
{
    FdeRuntimeConfig();

    int schema_version;
    std::size_t atom_count;
    std::vector<FragmentDefinition> fragments;
    std::vector<RuntimeStateDefinition> states;
    std::string active_state;
    std::string active_fragment;
    std::string active_density_path;
    std::vector<RuntimeArtifactPath> frozen_density_artifacts;
    std::vector<RuntimeArtifactPath> determinant_artifacts;
    std::vector<RuntimeArtifactPath> linearized_state_artifacts;
    std::vector<RuntimeDiagonalEnergy> diagonal_energies;
    std::string ao_overlap_path;
    std::string output_prefix;

    std::string fragment_xc;
    std::string embedding_xc;
    KineticFunctional kinetic_functional;
    double density_floor_bohr3;
    int maximum_scf_iterations;
    double scf_density_tolerance;
    double electron_tolerance;
    double mixing_beta;
    int maximum_freeze_thaw_cycles;
    double freeze_thaw_density_tolerance;
    double energy_tolerance_ry;
    std::vector<std::string> update_order;

    std::vector<std::string> k_state_labels;
    std::vector<std::string> l_fragment_labels;
    std::vector<std::string> m_fragment_labels;
    std::string coupling_provider;
    double transition_density_trace_tolerance;
    double singular_value_tolerance;
    double overlap_eigenvalue_cutoff;
    double symmetry_tolerance;
    double residual_tolerance;
    double minimum_root_overlap;
    bool calculate_force;
    double hartree_reciprocity_tolerance_ry;
};

class FdeRuntimeConfigIO
{
  public:
    static FdeRuntimeConfig read(std::istream& input);
    static void write(std::ostream& output, const FdeRuntimeConfig& config);
    static void validate(const FdeRuntimeConfig& config);
    static std::size_t state_index(const FdeRuntimeConfig& config,
                                   const std::string& label);
    static std::size_t fragment_index(const FdeRuntimeConfig& config,
                                      const std::string& label);
};

} // namespace fde

#endif // FDE_RUNTIME_CONFIG_H
