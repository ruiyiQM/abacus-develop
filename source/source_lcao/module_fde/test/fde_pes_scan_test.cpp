#include "../orchestration/fde_pes_scan.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

fde::FrozenDensityArtifact artifact(const fde::GeometryPoint& geometry,
                                    const fde::DiabaticStateDefinition& state,
                                    const bool first)
{
    fde::FrozenDensityArtifact value;
    value.schema_version = 1;
    value.fragment_label = first ? state.first_fragment_label : state.second_fragment_label;
    value.state_label = state.label;
    value.geometry_fingerprint = geometry.geometry_fingerprint;
    value.grid_fingerprint = "grid";
    value.pseudopotential_fingerprint = "pseudo";
    value.orbital_fingerprint = "orbital";
    value.core_density_fingerprint = value.fragment_label + "_core";
    value.xc_functional = "PBE";
    value.kinetic_functional = "PW91K";
    value.grid_x = 2;
    value.grid_y = 1;
    value.grid_z = 1;
    value.cell_volume_bohr3 = 2.0;
    value.alpha_electrons = first ? state.first_alpha_electrons
                                  : state.second_alpha_electrons;
    value.beta_electrons = first ? state.first_beta_electrons
                                 : state.second_beta_electrons;
    value.freeze_thaw_cycle = 2;
    value.scf_converged = true;
    value.orbital_kinetic_energy_ry = 0.0;
    value.nonlocal_pseudopotential_energy_ry = 0.0;
    value.rho_alpha_bohr3.assign(2, 0.5 * value.alpha_electrons);
    value.rho_beta_bohr3.assign(2, 0.5 * value.beta_electrons);
    return value;
}

class CrossingRunner : public fde::DiabaticStateRunner
{
  public:
    CrossingRunner() : localization(0.95), change_population(false) {}

    fde::DiabaticStateRunResult run(
        const fde::GeometryPoint& geometry,
        const fde::DiabaticStateDefinition& state,
        const fde::FreezeThawCheckpoint* warm_start) const override
    {
        if (warm_start != 0 && warm_start->first.state_label != state.label)
        {
            throw std::runtime_error("cross-state warm start");
        }
        warm_start_labels.push_back(warm_start == 0 ? "none" : warm_start->first.state_label);

        fde::DiabaticStateRunResult result;
        result.checkpoint.schema_version = 1;
        result.checkpoint.first = artifact(geometry, state, true);
        result.checkpoint.second = artifact(geometry, state, false);
        if (change_population)
        {
            result.checkpoint.first.alpha_electrons += 1;
            result.checkpoint.first.rho_alpha_bohr3.assign(
                2,
                0.5 * result.checkpoint.first.alpha_electrons);
        }
        const double energy = state.label == "donor" ? geometry.coordinate_bohr
                                                     : -geometry.coordinate_bohr;
        result.checkpoint.energy
            = {0.0, 0.0, 0.0, 0.0, 0.0, energy, 0.0, 0.0, 0.0};
        result.checkpoint.completed_cycles = 2;
        result.checkpoint.converged = true;
        result.checkpoint.density_residual = 0.0;
        result.checkpoint.energy_change_ry = 0.0;
        result.localization_score = localization;
        return result;
    }

    mutable std::vector<std::string> warm_start_labels;
    double localization;
    bool change_population;
};

std::vector<fde::GeometryPoint> geometries()
{
    return {{"g0", "hash0", -1.0},
            {"g1", "hash1", -0.25},
            {"g2", "hash2", 1.0}};
}

std::vector<fde::DiabaticStateDefinition> states()
{
    return {{"donor", "A", "B", 1, 0, 0, 1},
            {"acceptor", "A", "B", 0, 1, 1, 0}};
}

} // namespace

TEST(FdePesScan, KeepsIndependentWarmStartsAcrossACrossing)
{
    CrossingRunner runner;
    const fde::DiabaticPes result
        = fde::TwoStatePesScan::run(geometries(), states(), {1.0e-10, 0.9}, runner);

    ASSERT_EQ(result.points.size(), 6u);
    EXPECT_EQ(result.points[0].state_label, "donor");
    EXPECT_EQ(result.points[1].state_label, "acceptor");
    EXPECT_DOUBLE_EQ(result.points[4].energy_ry, 1.0);
    EXPECT_DOUBLE_EQ(result.points[5].energy_ry, -1.0);
    EXPECT_EQ(result.diagnostics.crossing_brackets, 1);
    EXPECT_DOUBLE_EQ(result.diagnostics.minimum_localization_score, 0.95);
    EXPECT_EQ(runner.warm_start_labels,
              (std::vector<std::string>{"none", "none", "donor", "acceptor",
                                        "donor", "acceptor"}));

    std::ostringstream first_output;
    std::ostringstream second_output;
    fde::TwoStatePesScan::write(first_output, result);
    fde::TwoStatePesScan::write(second_output, result);
    EXPECT_EQ(first_output.str(), second_output.str());
    EXPECT_NE(first_output.str().find("POINT g2 hash2 1 donor 1"), std::string::npos);
}

TEST(FdePesScan, RejectsLossOfDiabaticLocalization)
{
    CrossingRunner runner;
    runner.localization = 0.4;
    EXPECT_THROW(fde::TwoStatePesScan::run(geometries(), states(), {1.0e-10, 0.9}, runner),
                 std::runtime_error);
}

TEST(FdePesScan, RejectsPopulationDriftAndUnorderedCoordinates)
{
    CrossingRunner runner;
    runner.change_population = true;
    EXPECT_THROW(fde::TwoStatePesScan::run(geometries(), states(), {1.0e-10, 0.9}, runner),
                 std::invalid_argument);

    std::vector<fde::GeometryPoint> unordered = geometries();
    unordered[2].coordinate_bohr = -0.5;
    CrossingRunner ordered_runner;
    EXPECT_THROW(fde::TwoStatePesScan::run(unordered,
                                          states(),
                                          {1.0e-10, 0.9},
                                          ordered_runner),
                 std::invalid_argument);
}
