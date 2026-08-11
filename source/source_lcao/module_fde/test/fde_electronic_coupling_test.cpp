#include "../fde_electronic_coupling.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

fde::OccupiedSpinOrbitals occupied(const std::vector<double>& coefficient,
                                   const std::string& fragment)
{
    fde::OccupiedSpinOrbitals result;
    result.coefficients = coefficient;
    result.orbital_energies_ry = {-1.0};
    result.source_fragment_labels = {fragment};
    return result;
}

fde::DiabaticDeterminantArtifact determinant(const std::string& state,
                                             const std::vector<double>& alpha)
{
    fde::DiabaticDeterminantArtifact result;
    result.schema_version = 1;
    result.state_label = state;
    result.geometry_fingerprint = "geometry";
    result.orbital_fingerprint = "basis";
    result.ao_dimension = 2;
    result.alpha = occupied(alpha, "A");
    result.beta = occupied({0.0, 1.0}, "B");
    return result;
}

std::vector<double> identity_overlap()
{
    return {1.0, 0.0, 0.0, 1.0};
}

class DirectionalEnergy : public fde::TransitionEnergyProvider
{
  public:
    std::string name() const override
    {
        return "test_directional";
    }

    double evaluate_ry(const fde::DiabaticDeterminantArtifact& bra,
                       const fde::DiabaticDeterminantArtifact&,
                       const fde::SpinTransitionDensityMatrix& density,
                       const std::size_t ao_dimension) const override
    {
        EXPECT_NEAR(fde::ElectronicCoupling::electron_count(density.alpha,
                                                            identity_overlap(),
                                                            ao_dimension),
                    1.0,
                    1.0e-12);
        EXPECT_NEAR(fde::ElectronicCoupling::electron_count(density.beta,
                                                            identity_overlap(),
                                                            ao_dimension),
                    1.0,
                    1.0e-12);
        return bra.state_label == "first" ? 5.0 : 7.0;
    }
};

} // namespace

TEST(FdeElectronicCoupling, FormsNormalizedOverlapAndBidirectionalTransitionDensity)
{
    const fde::DiabaticDeterminantArtifact first
        = determinant("first", {1.0, 0.0});
    const fde::DiabaticDeterminantArtifact second
        = determinant("second", {0.8, 0.6});
    const fde::DeterminantTransition forward
        = fde::ElectronicCoupling::transition(first,
                                              second,
                                              identity_overlap(),
                                              1.0e-12);
    const fde::DeterminantTransition reverse
        = fde::ElectronicCoupling::transition(second,
                                              first,
                                              identity_overlap(),
                                              1.0e-12);

    EXPECT_NEAR(forward.normalized_overlap, 0.8, 1.0e-12);
    EXPECT_EQ(forward.density_matrix.alpha,
              (std::vector<double>{1.0, 0.75, 0.0, 0.0}));
    EXPECT_EQ(reverse.density_matrix.alpha,
              (std::vector<double>{1.0, 0.0, 0.75, 0.0}));
    EXPECT_NEAR(fde::ElectronicCoupling::electron_count(forward.density_matrix.alpha,
                                                        identity_overlap(),
                                                        2),
                1.0,
                1.0e-12);
}

TEST(FdeElectronicCoupling, SymmetrizesTransitionEnergyIntoHamiltonianCoupling)
{
    const DirectionalEnergy energy;
    const fde::CouplingValidationControls validation{1.0e-12, 1.0e-12};
    const fde::ElectronicCouplingResult result
        = fde::ElectronicCoupling::evaluate_symmetric(
            determinant("first", {1.0, 0.0}),
            determinant("second", {0.8, 0.6}),
            identity_overlap(),
            energy,
            validation,
            1.0e-12);

    EXPECT_NEAR(result.normalized_overlap, 0.8, 1.0e-12);
    EXPECT_DOUBLE_EQ(result.forward_transition_energy_ry, 5.0);
    EXPECT_DOUBLE_EQ(result.reverse_transition_energy_ry, 7.0);
    EXPECT_NEAR(result.hamiltonian_coupling_ry, 4.8, 1.0e-12);
    EXPECT_EQ(result.provider, "test_directional");
    EXPECT_NEAR(result.transition_energy_asymmetry_ry, 2.0, 1.0e-12);
    EXPECT_NEAR(result.estimated_coupling_uncertainty_ry, 0.8, 1.0e-12);
    EXPECT_LE(result.maximum_transition_density_trace_error, 1.0e-12);
}

TEST(FdeElectronicCoupling, RejectsSingularOccupiedOverlap)
{
    EXPECT_THROW(fde::ElectronicCoupling::transition(
                     determinant("first", {1.0, 0.0}),
                     determinant("second", {0.0, 1.0}),
                     identity_overlap(),
                     1.0e-12),
                 std::runtime_error);
}
