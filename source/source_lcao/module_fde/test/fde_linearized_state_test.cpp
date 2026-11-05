#include "../coupling/fde_linearized_state.h"
#include "../coupling/fde_multistate_solver.h"

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <vector>

namespace
{

fde::DiabaticDeterminantArtifact determinant(const std::string& label,
                                             const std::vector<double>& alpha)
{
    fde::DiabaticDeterminantArtifact result;
    result.schema_version = 1;
    result.state_label = label;
    result.geometry_fingerprint = "geometry";
    result.orbital_fingerprint = "basis";
    result.ao_dimension = 2;
    result.alpha.coefficients = alpha;
    result.alpha.orbital_energies_ry = {-1.0};
    result.alpha.source_fragment_labels = {"F"};
    result.beta.coefficients = {0.0, 1.0};
    result.beta.orbital_energies_ry = {-0.5};
    result.beta.source_fragment_labels = {"CH3Cl"};
    return result;
}

fde::LinearizedStateArtifact state(const std::string& label, const double energy)
{
    fde::LinearizedStateArtifact result;
    result.schema_version = 1;
    result.state_label = label;
    result.geometry_fingerprint = "geometry";
    result.ao_dimension = 2;
    result.reference_energy_ry = energy;
    result.hamiltonian_alpha_ry = {2.0, 0.5, 0.5, 1.0};
    result.hamiltonian_beta_ry = {1.0, 0.0, 0.0, 3.0};
    return result;
}

} // namespace

TEST(FdeLinearizedState, RoundTripsArtifact)
{
    std::ostringstream output;
    fde::LinearizedStateArtifactIO::write(output, state("first", -10.0), 1.0e-12);
    std::istringstream input(output.str());
    const fde::LinearizedStateArtifact copy
        = fde::LinearizedStateArtifactIO::read(input, 1.0e-12);
    EXPECT_EQ(copy.state_label, "first");
    EXPECT_DOUBLE_EQ(copy.reference_energy_ry, -10.0);
    EXPECT_DOUBLE_EQ(copy.hamiltonian_alpha_ry[1], 0.5);
}

TEST(FdeLinearizedState, ReproducesReferenceAndDirectionalFirstOrderEnergy)
{
    const std::vector<double> overlap = {1.0, 0.0, 0.0, 1.0};
    const fde::DiabaticDeterminantArtifact first
        = determinant("first", {1.0, 0.0});
    const fde::DiabaticDeterminantArtifact second
        = determinant("second", {0.8, 0.6});
    const fde::LinearizedTransitionEnergy energy(
        {state("first", -10.0), state("second", -9.0)}, overlap, 1.0e-12);
    const fde::DeterminantTransition self
        = fde::ElectronicCoupling::transition(first, first, overlap, 1.0e-12);
    EXPECT_NEAR(energy.evaluate_ry(first, first, self.density_matrix, 2),
                -10.0,
                1.0e-12);
    const fde::DeterminantTransition forward
        = fde::ElectronicCoupling::transition(first, second, overlap, 1.0e-12);
    EXPECT_NE(energy.evaluate_ry(first, second, forward.density_matrix, 2),
              -10.0);
}

TEST(FdeLinearizedState, RecoversAnalyticOneBodyRootsUnderOrbitalSignChange)
{
    const std::vector<double> overlap = {1.0, 0.0, 0.0, 1.0};
    const fde::DiabaticDeterminantArtifact first = determinant("first", {1.0, 0.0});
    const std::vector<fde::LinearizedStateArtifact> states
        = {state("first", 5.0), state("second", 5.12)};
    const fde::LinearizedTransitionEnergy energy(states, overlap, 1.0e-12);
    const fde::CouplingValidationControls validation{1.0e-12, 1.0e-12};
    const fde::NonorthogonalSolverControls controls{1.0e-10, 1.0e-12, 1.0e-10};
    // One alpha electron in [[2,.5],[.5,1]], plus a beta spectator of energy 3.
    const double lower = 4.5 - std::sqrt(0.5);
    const double upper = 4.5 + std::sqrt(0.5);
    for (const double sign : {-1.0, 1.0})
    {
        fde::DiabaticDeterminantArtifact second = determinant("second", {0.8, 0.6});
        for (double& coefficient : second.alpha.coefficients)
        {
            coefficient *= sign;
        }
        const fde::ElectronicCouplingResult coupling = fde::ElectronicCoupling::evaluate_symmetric(
            first, second, overlap, energy, validation, 1.0e-12);
        const double s = coupling.normalized_overlap;
        const double h = coupling.hamiltonian_coupling_ry;
        const double expected_s = sign * 0.8;
        const double expected_h = sign * 4.3;
        EXPECT_NEAR(s, expected_s, 1.0e-12);
        EXPECT_NEAR(h, expected_h, 1.0e-12);
        const double signed_coupling = (h - 5.06 * s) / (1.0 - s * s);
        const double effective_coupling = std::fabs(signed_coupling);
        EXPECT_NEAR(effective_coupling, 0.7, 1.0e-12);
        const fde::NonorthogonalStateProblem problem{{"first", "second"},
                                                   {5.0, h, h, 5.12},
                                                   {1.0, s, s, 1.0}};
        const fde::NonorthogonalStateSolution roots
            = fde::NonorthogonalMultistateSolver::solve(problem, controls);
        ASSERT_EQ(roots.eigenvalues_ry.size(), 2);
        EXPECT_NEAR(roots.eigenvalues_ry[0], lower, 1.0e-12);
        EXPECT_NEAR(roots.eigenvalues_ry[1], upper, 1.0e-12);
        EXPECT_LE(coupling.maximum_transition_density_trace_error, 1.0e-12);
        EXPECT_LE(coupling.overlap_reciprocity_error, 1.0e-12);
    }
}
