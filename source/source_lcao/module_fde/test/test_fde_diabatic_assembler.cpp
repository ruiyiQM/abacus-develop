#include "../coupling/fde_diabatic_assembler.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{

using Complex = std::complex<double>;

fde::ComplexDiabaticDeterminantArtifact state(const std::string& label,
                                             const std::size_t site)
{
    fde::ComplexDiabaticDeterminantArtifact value;
    value.schema_version = 2;
    value.state_label = label;
    value.geometry_fingerprint = "fixed_cell";
    value.orbital_fingerprint = "fixed_basis";
    value.resource_fingerprint = "fixed_resources";
    value.boundary_twist = {0.17, 0.11, 0.0};
    value.ao_dimension = 3;
    value.alpha.coefficients.assign(3, 0.0);
    value.alpha.coefficients[site] = 1.0;
    value.alpha.orbital_energies_ry = {-1.0};
    const std::string source = site == 0 ? "D" : "A";
    value.alpha.source_fragment_labels = {source};
    value.beta.coefficients = {0.0, 0.0, 1.0};
    value.beta.orbital_energies_ry = {-2.0};
    value.beta.source_fragment_labels = {"A"};
    return value;
}

class ConstantEnergy : public fde::ComplexTransitionEnergyProvider
{
  public:
    std::string name() const override { return "constant_two"; }
    Complex evaluate_ry(const fde::ComplexDiabaticDeterminantArtifact&,
                        const fde::ComplexDiabaticDeterminantArtifact&,
                        const fde::ComplexSpinTransitionDensityMatrix&,
                        const std::size_t) const override
    {
        return 2.0;
    }
};

} // namespace

TEST(FdeComplexDiabaticAssembler, RetainsFullOccupiedOverlapAndBuildsHermitianStateMatrices)
{
    const std::vector<fde::ComplexDiabaticDeterminantArtifact> states = {state("D", 0), state("A", 1)};
    const std::vector<Complex> metric = {1.0, Complex(0.0, -0.8), 0.0,
                                        Complex(0.0, 0.8), 1.0, 0.0,
                                        0.0, 0.0, 1.0};
    const std::vector<double> energies = {2.0, 3.0};
    const fde::FdeDiabApproximationSpec selection{{0, 1}, {"D", "A"}, {"D", "A"}};
    const fde::CouplingValidationControls controls{1.0e-12, 1.0e-12};
    const ConstantEnergy provider;
    const auto result = fde::ComplexFdeDiabaticAssembler::assemble(
        states, energies, metric, selection, provider, controls, 1.0e-12);
    EXPECT_EQ(result.problem.overlap[2], Complex(0.0, 0.8));
    EXPECT_EQ(result.problem.overlap[1], Complex(0.0, -0.8));
    EXPECT_EQ(result.problem.hamiltonian_ry[2], Complex(0.0, 1.6));
    EXPECT_EQ(result.problem.hamiltonian_ry[1], Complex(0.0, -1.6));
    EXPECT_EQ(result.problem.hamiltonian_ry[0], Complex(2.0, 0.0));
    EXPECT_EQ(result.problem.hamiltonian_ry[3], Complex(3.0, 0.0));
    ASSERT_EQ(result.pairs.size(), 1);
    EXPECT_LE(result.pairs[0].maximum_transition_density_trace_error, 1.0e-12);
    EXPECT_EQ(result.audit.retained_interfragment_overlap_pairs, 1);
    EXPECT_EQ(result.audit.discarded_interfragment_overlap_pairs, 0);
}
