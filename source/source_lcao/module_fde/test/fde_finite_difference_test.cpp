#include "../fde_finite_difference.h"

#include <gtest/gtest.h>

#include <cmath>
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
    value.freeze_thaw_cycle = 3;
    value.scf_converged = true;
    value.orbital_kinetic_energy_ry = 0.0;
    value.nonlocal_pseudopotential_energy_ry = 0.0;
    value.rho_alpha_bohr3.assign(2, 0.5 * value.alpha_electrons);
    value.rho_beta_bohr3.assign(2, 0.5 * value.beta_electrons);
    return value;
}

fde::FreezeThawCheckpoint checkpoint(const fde::GeometryPoint& geometry,
                                     const fde::DiabaticStateDefinition& state,
                                     const double energy,
                                     const bool converged)
{
    fde::FreezeThawCheckpoint value;
    value.schema_version = 1;
    value.first = artifact(geometry, state, true);
    value.second = artifact(geometry, state, false);
    value.energy = {0.0, 0.0, 0.0, 0.0, 0.0, energy, 0.0, 0.0, 0.0};
    value.completed_cycles = 3;
    value.converged = converged;
    value.density_residual = 0.0;
    value.energy_change_ry = 0.0;
    return value;
}

class PolynomialRunner : public fde::DiabaticStateRunner
{
  public:
    PolynomialRunner() : quartic(false), fail_label("") {}

    fde::DiabaticStateRunResult run(
        const fde::GeometryPoint& geometry,
        const fde::DiabaticStateDefinition& state,
        const fde::FreezeThawCheckpoint* warm_start) const override
    {
        if (warm_start == 0 || warm_start->first.state_label != state.label)
        {
            throw std::runtime_error("missing or cross-state reference checkpoint");
        }
        reference_fingerprints.push_back(warm_start->first.geometry_fingerprint);
        const double coordinate = geometry.coordinate_bohr;
        const double energy = quartic ? std::pow(coordinate, 4.0)
                                      : coordinate * coordinate;
        return {checkpoint(geometry, state, energy, geometry.label != fail_label), 0.98};
    }

    mutable std::vector<std::string> reference_fingerprints;
    bool quartic;
    std::string fail_label;
};

fde::GeometryPoint geometry(const std::string& label, const double coordinate)
{
    return {label, label + "_hash", coordinate};
}

fde::FiniteDifferenceStencil stencil()
{
    return {"atom0_x",
            geometry("reference", 1.0),
            geometry("minus", 0.8),
            geometry("minus_half", 0.9),
            geometry("plus_half", 1.1),
            geometry("plus", 1.2)};
}

fde::DiabaticStateDefinition state(const std::string& label)
{
    return {label, "A", "B", 1, 0, 0, 1};
}

fde::FiniteDifferenceControls controls(const double force_tolerance)
{
    return {0.2, 1.0e-12, 1.0e-10, 0.9, force_tolerance};
}

} // namespace

TEST(FdeFiniteDifference, ReconvergesEveryDisplacementFromOneStateReference)
{
    PolynomialRunner runner;
    const fde::DiabaticStateDefinition donor = state("donor");
    const fde::FiniteDifferenceForceResult result
        = fde::FiniteDifferenceForce::evaluate(
            stencil(),
            donor,
            checkpoint(stencil().reference, donor, 1.0, true),
            controls(1.0e-10),
            runner);

    EXPECT_NEAR(result.coarse_force_ry_per_bohr, -2.0, 1.0e-12);
    EXPECT_NEAR(result.fine_force_ry_per_bohr, -2.0, 1.0e-12);
    EXPECT_NEAR(result.richardson_force_ry_per_bohr, -2.0, 1.0e-12);
    EXPECT_NEAR(result.estimated_error_ry_per_bohr, 0.0, 1.0e-12);
    EXPECT_EQ(result.total_freeze_thaw_cycles, 12);
    EXPECT_EQ(runner.reference_fingerprints,
              (std::vector<std::string>{"reference_hash", "reference_hash",
                                        "reference_hash", "reference_hash"}));
}

TEST(FdeFiniteDifference, KeepsTheTwoDiabaticStatesIndependent)
{
    PolynomialRunner runner;
    const fde::DiabaticStateDefinition donor = state("donor");
    const fde::DiabaticStateDefinition acceptor = state("acceptor");
    const fde::FiniteDifferenceForceResult donor_result
        = fde::FiniteDifferenceForce::evaluate(
            stencil(),
            donor,
            checkpoint(stencil().reference, donor, 1.0, true),
            controls(1.0e-10),
            runner);
    const fde::FiniteDifferenceForceResult acceptor_result
        = fde::FiniteDifferenceForce::evaluate(
            stencil(),
            acceptor,
            checkpoint(stencil().reference, acceptor, 1.0, true),
            controls(1.0e-10),
            runner);

    EXPECT_EQ(donor_result.state_label, "donor");
    EXPECT_EQ(acceptor_result.state_label, "acceptor");
}

TEST(FdeFiniteDifference, RejectsUnconvergedAndStepDependentDisplacements)
{
    PolynomialRunner unconverged_runner;
    unconverged_runner.fail_label = "plus";
    const fde::DiabaticStateDefinition donor = state("donor");
    EXPECT_THROW(fde::FiniteDifferenceForce::evaluate(
                     stencil(),
                     donor,
                     checkpoint(stencil().reference, donor, 1.0, true),
                     controls(1.0),
                     unconverged_runner),
                 std::runtime_error);

    PolynomialRunner step_dependent_runner;
    step_dependent_runner.quartic = true;
    EXPECT_THROW(fde::FiniteDifferenceForce::evaluate(
                     stencil(),
                     donor,
                     checkpoint(stencil().reference, donor, 1.0, true),
                     controls(1.0e-3),
                     step_dependent_runner),
                 std::runtime_error);
}
