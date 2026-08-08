#include "../fde_multi_fragment_freeze_thaw.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

fde::FrozenDensityArtifact artifact(const std::string& fragment,
                                    const std::vector<double>& alpha,
                                    const std::vector<double>& beta)
{
    fde::FrozenDensityArtifact value;
    value.schema_version = 1;
    value.fragment_label = fragment;
    value.state_label = "state_1";
    value.geometry_fingerprint = "geometry";
    value.grid_fingerprint = "grid";
    value.pseudopotential_fingerprint = "pseudo";
    value.orbital_fingerprint = "orbital";
    value.core_density_fingerprint = fragment + "_core";
    value.xc_functional = "PBE";
    value.kinetic_functional = "PW91K";
    value.grid_x = 2;
    value.grid_y = 1;
    value.grid_z = 1;
    value.cell_volume_bohr3 = 2.0;
    value.alpha_electrons = 1;
    value.beta_electrons = 1;
    value.freeze_thaw_cycle = 0;
    value.scf_converged = true;
    value.orbital_kinetic_energy_ry = 0.1;
    value.nonlocal_pseudopotential_energy_ry = -0.2;
    value.rho_alpha_bohr3 = alpha;
    value.rho_beta_bohr3 = beta;
    return value;
}

class ThreeFragmentRunner : public fde::MultiFragmentFreezeThawStepRunner
{
  public:
    fde::FrozenDensityArtifact update(
        const fde::FrozenDensityArtifact& active,
        const std::vector<fde::FrozenDensityArtifact>& frozen_environment) const override
    {
        EXPECT_EQ(frozen_environment.size(), 2);
        for (std::size_t index = 0; index < frozen_environment.size(); ++index)
        {
            EXPECT_NE(frozen_environment[index].fragment_label, active.fragment_label);
        }
        calls.push_back(active.fragment_label);
        fde::FrozenDensityArtifact updated = active;
        if (active.fragment_label == "A")
        {
            updated.rho_alpha_bohr3 = {0.6, 0.4};
            updated.rho_beta_bohr3 = {0.6, 0.4};
        }
        else if (active.fragment_label == "B")
        {
            updated.rho_alpha_bohr3 = {0.5, 0.5};
            updated.rho_beta_bohr3 = {0.5, 0.5};
        }
        else
        {
            updated.rho_alpha_bohr3 = {0.4, 0.6};
            updated.rho_beta_bohr3 = {0.4, 0.6};
        }
        ++updated.freeze_thaw_cycle;
        return updated;
    }

    fde::MultiFragmentEnergyLedger canonical_energy(
        const std::vector<fde::FrozenDensityArtifact>& fragments) const override
    {
        fde::MultiFragmentEnergyLedger ledger;
        ledger.nonadditive_kinetic_ry = 0.0;
        ledger.total_hartree_ry = 0.0;
        ledger.total_xc_ry = 0.0;
        ledger.local_external_ry = 0.0;
        ledger.ion_ion_ry = 0.0;
        for (std::size_t fragment = 0; fragment < fragments.size(); ++fragment)
        {
            fde::FragmentEnergyTerm term;
            term.fragment_label = fragments[fragment].fragment_label;
            term.orbital_kinetic_ry = fragments[fragment].orbital_kinetic_energy_ry;
            term.nonlocal_external_ry
                = fragments[fragment].nonlocal_pseudopotential_energy_ry;
            ledger.fragments.push_back(term);
            for (std::size_t index = 0;
                 index < fragments[fragment].rho_alpha_bohr3.size();
                 ++index)
            {
                ledger.total_hartree_ry
                    += fragments[fragment].rho_alpha_bohr3[index]
                       * fragments[fragment].rho_alpha_bohr3[index];
            }
        }
        return ledger;
    }

    mutable std::vector<std::string> calls;
};

std::vector<fde::FrozenDensityArtifact> initial_fragments()
{
    return {artifact("A", {0.8, 0.2}, {0.8, 0.2}),
            artifact("B", {0.7, 0.3}, {0.7, 0.3}),
            artifact("C", {0.2, 0.8}, {0.2, 0.8})};
}

fde::MultiFragmentFreezeThawControls controls(const int maximum_cycles)
{
    return {maximum_cycles, 1.0e-12, 1.0e-12, 1.0e-10, {2, 0, 1}};
}

} // namespace

TEST(FdeMultiFragmentFreezeThaw, RunsExplicitPermutationForEveryCompleteCycle)
{
    ThreeFragmentRunner runner;
    const fde::MultiFragmentFreezeThawCheckpoint initial
        = fde::MultiFragmentFreezeThawWorkflow::initialize(initial_fragments(),
                                                           runner,
                                                           1.0e-10);
    const fde::MultiFragmentFreezeThawCheckpoint result
        = fde::MultiFragmentFreezeThawWorkflow::run(initial, controls(3), runner);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.completed_cycles, 2);
    EXPECT_EQ(runner.calls,
              (std::vector<std::string>{"C", "A", "B", "C", "A", "B"}));
    EXPECT_DOUBLE_EQ(result.density_residual, 0.0);
    EXPECT_EQ(result.energy.fragments.size(), 3);
}

TEST(FdeMultiFragmentFreezeThaw, VariableLengthCheckpointRoundTripsAndResumes)
{
    ThreeFragmentRunner runner;
    const fde::MultiFragmentFreezeThawCheckpoint initial
        = fde::MultiFragmentFreezeThawWorkflow::initialize(initial_fragments(),
                                                           runner,
                                                           1.0e-10);
    const fde::MultiFragmentFreezeThawCheckpoint first_cycle
        = fde::MultiFragmentFreezeThawWorkflow::run(initial, controls(1), runner);
    ASSERT_FALSE(first_cycle.converged);

    std::ostringstream output;
    fde::MultiFragmentFreezeThawWorkflow::write_checkpoint(output, first_cycle);
    std::istringstream input(output.str());
    const fde::MultiFragmentFreezeThawCheckpoint restored
        = fde::MultiFragmentFreezeThawWorkflow::read_checkpoint(input, 1.0e-10);
    const fde::MultiFragmentFreezeThawCheckpoint result
        = fde::MultiFragmentFreezeThawWorkflow::run(restored, controls(3), runner);

    EXPECT_TRUE(result.converged);
    ASSERT_EQ(result.fragments.size(), 3);
    EXPECT_EQ(result.fragments[2].fragment_label, "C");
    EXPECT_DOUBLE_EQ(fde::MultiFragmentCanonicalEnergy::total(result.energy),
                     fde::MultiFragmentCanonicalEnergy::total(restored.energy));
}

TEST(FdeMultiFragmentFreezeThaw, RejectsAnIncompleteUpdatePermutation)
{
    ThreeFragmentRunner runner;
    const fde::MultiFragmentFreezeThawCheckpoint initial
        = fde::MultiFragmentFreezeThawWorkflow::initialize(initial_fragments(),
                                                           runner,
                                                           1.0e-10);
    fde::MultiFragmentFreezeThawControls invalid = controls(2);
    invalid.update_order = {0, 1, 1};

    EXPECT_THROW(fde::MultiFragmentFreezeThawWorkflow::run(initial, invalid, runner),
                 std::invalid_argument);
}
