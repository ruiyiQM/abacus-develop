#include "../io/fde_fragment_artifact.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

namespace
{

using Complex = std::complex<double>;

fde::ComplexFragmentScfArtifact fragment()
{
    fde::ComplexFragmentScfArtifact value;
    value.schema_version = 2;
    value.state_label = "D_hole";
    value.fragment_label = "D";
    value.geometry_fingerprint = "fixed_cell";
    value.orbital_fingerprint = "fixed_basis";
    value.boundary_twist = {0.17, 0.11, 0.0};
    value.resource_fingerprint = "fixed_pp_ao_grid_functionals";
    value.density_path = "density.fde";
    value.freeze_thaw_cycle = 2;
    value.scf_converged = true;
    value.ao_dimension = 2;
    value.active_orbitals = {0};
    value.alpha.coefficients = {Complex(0.0, 1.0), 0.0};
    value.alpha.orbital_energies_ry = {-1.0};
    value.alpha.source_fragment_labels = {"D"};
    value.beta = value.alpha;
    value.ao_overlap = {1.0, Complex(0.0, -0.1), Complex(0.0, 0.1), 1.0};
    value.hamiltonian_alpha_ry = {-1.0, Complex(0.2, -0.3), Complex(0.2, 0.3), -0.5};
    value.hamiltonian_beta_ry = value.hamiltonian_alpha_ry;
    value.subsystem_total_energy_ry = -5.0;
    value.ion_ion_energy_ry = 1.0;
    value.hartree_cross_energy_ry = 0.2;
    value.nonadditive_kinetic_energy_ry = 0.1;
    value.nonadditive_xc_energy_ry = -0.01;
    return value;
}

} // namespace

TEST(FdeComplexFragmentArtifact, RoundTripsComplexOccupiedMetricHamiltonianAndBoundary)
{
    const auto value = fragment();
    std::ostringstream output;
    fde::ComplexFragmentScfArtifactIO::write(output, value, 1.0e-12);
    std::istringstream input(output.str());
    const auto restored = fde::ComplexFragmentScfArtifactIO::read(input, 1.0e-12);
    EXPECT_EQ(restored.alpha.coefficients, value.alpha.coefficients);
    EXPECT_EQ(restored.ao_overlap, value.ao_overlap);
    EXPECT_EQ(restored.hamiltonian_alpha_ry, value.hamiltonian_alpha_ry);
    EXPECT_EQ(restored.boundary_twist, value.boundary_twist);
    EXPECT_EQ(restored.resource_fingerprint, value.resource_fingerprint);
    EXPECT_EQ(restored.active_orbitals, value.active_orbitals);
}

TEST(FdeComplexFragmentArtifact, RejectsWrongBoundaryNonHermitianAndPartialOutputs)
{
    auto value = fragment();
    value.boundary_twist.clear();
    EXPECT_THROW(fde::ComplexFragmentScfArtifactIO::validate(value, 1.0e-12), std::invalid_argument);
    value = fragment();
    value.hamiltonian_alpha_ry[1] = value.hamiltonian_alpha_ry[2];
    EXPECT_THROW(fde::ComplexFragmentScfArtifactIO::validate(value, 1.0e-12), std::invalid_argument);
    value = fragment();
    value.scf_converged = false;
    EXPECT_THROW(fde::ComplexFragmentScfArtifactIO::validate(value, 1.0e-12), std::invalid_argument);
}

TEST(FdeComplexFragmentArtifact, RejectsOccupiedCoefficientsOutsideOwnedAoSpace)
{
    auto value = fragment();
    value.alpha.coefficients[1] = Complex(0.0, 0.1);
    EXPECT_THROW(fde::ComplexFragmentScfArtifactIO::validate(value, 1.0e-12), std::invalid_argument);
}

TEST(FdeComplexFragmentArtifact, RejectsWrongScalarFormatAndTruncatedComplexData)
{
    const auto value = fragment();
    std::ostringstream output;
    fde::ComplexFragmentScfArtifactIO::write(output, value, 1.0e-12);
    std::istringstream wrong_scalar(output.str());
    EXPECT_THROW(fde::FragmentScfArtifactIO::read(wrong_scalar, 1.0e-12), std::invalid_argument);
    std::string malformed = output.str();
    const auto position = malformed.find('(');
    ASSERT_NE(position, std::string::npos);
    malformed.resize(position + 2);
    std::istringstream truncated(malformed);
    EXPECT_THROW(fde::ComplexFragmentScfArtifactIO::read(truncated, 1.0e-12), std::invalid_argument);
}
