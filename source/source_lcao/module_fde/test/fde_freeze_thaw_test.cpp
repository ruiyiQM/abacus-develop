#include "../fde_freeze_thaw.h"

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
    value.orbital_kinetic_energy_ry = 0.0;
    value.nonlocal_pseudopotential_energy_ry = 0.0;
    value.rho_alpha_bohr3 = alpha;
    value.rho_beta_bohr3 = beta;
    return value;
}

class TargetStepRunner : public fde::FreezeThawStepRunner
{
  public:
    TargetStepRunner() : fail_updates_(false) {}

    fde::FrozenDensityArtifact update(
        const fde::FrozenDensityArtifact& active,
        const fde::FrozenDensityArtifact&) const override
    {
        calls.push_back(active.fragment_label);
        fde::FrozenDensityArtifact updated = active;
        const std::vector<double> target = active.fragment_label == "A"
                                               ? std::vector<double>{0.6, 0.4}
                                               : std::vector<double>{0.4, 0.6};
        updated.rho_alpha_bohr3 = target;
        updated.rho_beta_bohr3 = target;
        ++updated.freeze_thaw_cycle;
        updated.scf_converged = !fail_updates_;
        return updated;
    }

    fde::EnergyLedger canonical_energy(
        const fde::FrozenDensityArtifact& first,
        const fde::FrozenDensityArtifact& second) const override
    {
        double density_norm = 0.0;
        for (std::size_t index = 0; index < first.rho_alpha_bohr3.size(); ++index)
        {
            density_norm += first.rho_alpha_bohr3[index] * first.rho_alpha_bohr3[index]
                            + second.rho_alpha_bohr3[index] * second.rho_alpha_bohr3[index];
        }
        return {0.0, 0.0, 0.0, density_norm, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    mutable std::vector<std::string> calls;
    bool fail_updates_;
};

fde::FreezeThawControls controls(const int maximum_cycles,
                                 const fde::FreezeThawOrder order)
{
    return {maximum_cycles, 1.0e-12, 1.0e-12, 1.0e-10, order};
}

} // namespace

TEST(FdeFreezeThaw, AlternatesSubsystemsAndConvergesOnBothThresholds)
{
    TargetStepRunner runner;
    const fde::FreezeThawCheckpoint initial
        = fde::FreezeThawWorkflow::initialize(artifact("A", {0.8, 0.2}, {0.8, 0.2}),
                                              artifact("B", {0.2, 0.8}, {0.2, 0.8}),
                                              runner,
                                              1.0e-10);
    const fde::FreezeThawCheckpoint result
        = fde::FreezeThawWorkflow::run(initial,
                                      controls(3, fde::FreezeThawOrder::FirstThenSecond),
                                      runner);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.completed_cycles, 2);
    EXPECT_DOUBLE_EQ(result.density_residual, 0.0);
    EXPECT_DOUBLE_EQ(result.energy_change_ry, 0.0);
    EXPECT_EQ(runner.calls,
              (std::vector<std::string>{"A", "B", "A", "B"}));
}

TEST(FdeFreezeThaw, CheckpointRoundTripResumesWithoutChangingState)
{
    TargetStepRunner runner;
    const fde::FreezeThawCheckpoint initial
        = fde::FreezeThawWorkflow::initialize(artifact("A", {0.8, 0.2}, {0.8, 0.2}),
                                              artifact("B", {0.2, 0.8}, {0.2, 0.8}),
                                              runner,
                                              1.0e-10);
    const fde::FreezeThawCheckpoint first_cycle
        = fde::FreezeThawWorkflow::run(initial,
                                      controls(1, fde::FreezeThawOrder::SecondThenFirst),
                                      runner);
    ASSERT_FALSE(first_cycle.converged);

    std::ostringstream output;
    fde::FreezeThawWorkflow::write_checkpoint(output, first_cycle);
    std::istringstream input(output.str());
    const fde::FreezeThawCheckpoint restored
        = fde::FreezeThawWorkflow::read_checkpoint(input, 1.0e-10);
    const fde::FreezeThawCheckpoint result
        = fde::FreezeThawWorkflow::run(restored,
                                      controls(3, fde::FreezeThawOrder::SecondThenFirst),
                                      runner);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.completed_cycles, 2);
    EXPECT_EQ(result.first.rho_alpha_bohr3, first_cycle.first.rho_alpha_bohr3);
    EXPECT_EQ(result.second.rho_alpha_bohr3, first_cycle.second.rho_alpha_bohr3);
}

TEST(FdeFreezeThaw, UpdateOrdersReachTheSameConvergedEnergy)
{
    TargetStepRunner first_runner;
    TargetStepRunner second_runner;
    const fde::FrozenDensityArtifact first = artifact("A", {0.8, 0.2}, {0.8, 0.2});
    const fde::FrozenDensityArtifact second = artifact("B", {0.2, 0.8}, {0.2, 0.8});
    const fde::FreezeThawCheckpoint first_initial
        = fde::FreezeThawWorkflow::initialize(first, second, first_runner, 1.0e-10);
    const fde::FreezeThawCheckpoint second_initial
        = fde::FreezeThawWorkflow::initialize(first, second, second_runner, 1.0e-10);

    const fde::FreezeThawCheckpoint first_result
        = fde::FreezeThawWorkflow::run(first_initial,
                                      controls(3, fde::FreezeThawOrder::FirstThenSecond),
                                      first_runner);
    const fde::FreezeThawCheckpoint second_result
        = fde::FreezeThawWorkflow::run(second_initial,
                                      controls(3, fde::FreezeThawOrder::SecondThenFirst),
                                      second_runner);

    EXPECT_TRUE(first_result.converged);
    EXPECT_TRUE(second_result.converged);
    EXPECT_DOUBLE_EQ(fde::CanonicalEnergy::total(first_result.energy),
                     fde::CanonicalEnergy::total(second_result.energy));
    EXPECT_EQ(first_result.first.rho_alpha_bohr3, second_result.first.rho_alpha_bohr3);
    EXPECT_EQ(first_result.second.rho_alpha_bohr3, second_result.second.rho_alpha_bohr3);
}

TEST(FdeFreezeThaw, RejectsAnUnconvergedInnerUpdate)
{
    TargetStepRunner runner;
    const fde::FreezeThawCheckpoint initial
        = fde::FreezeThawWorkflow::initialize(artifact("A", {0.8, 0.2}, {0.8, 0.2}),
                                              artifact("B", {0.2, 0.8}, {0.2, 0.8}),
                                              runner,
                                              1.0e-10);
    runner.fail_updates_ = true;

    EXPECT_THROW(fde::FreezeThawWorkflow::run(
                     initial,
                     controls(2, fde::FreezeThawOrder::FirstThenSecond),
                     runner),
                 std::runtime_error);
}
