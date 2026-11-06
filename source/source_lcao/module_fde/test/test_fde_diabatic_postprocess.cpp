#include "../coupling/fde_diabatic_postprocess.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace
{

using Complex = std::complex<double>;

fde::FdeRuntimeConfig config()
{
    fde::FdeRuntimeConfig value;
    value.atom_count = 2;
    value.fragments = {{"D", {0}, 1}, {"A", {1}, 2}};
    // One alpha electron moves D -> A while a beta spectator remains in A.
    // Both states have N=2, M=0 and one hole relative to three neutral-valence electrons.
    fde::RuntimeStateDefinition first;
    first.state.label = "D";
    first.total_charge = 1;
    first.total_spin_projection = 0;
    first.state.fragments = {{"D", 0, 1}, {"A", 1, -1}};
    auto second = first;
    second.state.label = "A";
    second.state.fragments = {{"D", 1, 0}, {"A", 0, 0}};
    value.states = {first, second};
    value.diagonal_energies = {{"D", 5.0}, {"A", 4.4}};
    value.boundary_twist = {0.17, 0.11, 0.0};
    value.output_prefix = "unused";
    value.update_order = {"D", "A"};
    value.k_state_labels = {"D", "A"};
    value.l_fragment_labels = {"D", "A"};
    value.m_fragment_labels = {"D", "A"};
    return value;
}

std::vector<Complex> metric()
{
    return {1.0, Complex(0.0, -0.8), 0.0,
            Complex(0.0, 0.8), 1.0, 0.0,
            0.0, 0.0, 1.0};
}

fde::ComplexDiabaticDeterminantArtifact determinant(const std::string& label,
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
    value.alpha.source_fragment_labels = {label};
    value.beta.coefficients = {0.0, 0.0, 1.0};
    value.beta.orbital_energies_ry = {-2.0};
    value.beta.source_fragment_labels = {"A"};
    return value;
}

fde::ComplexLinearizedStateArtifact linearized(const std::string& label, const double energy)
{
    fde::ComplexLinearizedStateArtifact value;
    value.schema_version = 2;
    value.state_label = label;
    value.geometry_fingerprint = "fixed_cell";
    value.orbital_fingerprint = "fixed_basis";
    value.resource_fingerprint = "fixed_resources";
    value.boundary_twist = {0.17, 0.11, 0.0};
    value.ao_dimension = 3;
    value.reference_energy_ry = energy;
    value.hamiltonian_alpha_ry = {2.0, Complex(-0.3, -1.45), 0.0,
                                 Complex(-0.3, 1.45), 1.4, 0.0,
                                 0.0, 0.0, 0.0};
    value.hamiltonian_beta_ry = {3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0};
    return value;
}

} // namespace

TEST(FdeComplexDiabaticPostprocess, PreservesLocalElectronSectorAndAnalyticCouplingRoots)
{
    const auto runtime = config();
    const std::vector<fde::ComplexDiabaticDeterminantArtifact> states = {
        determinant("D", 0), determinant("A", 1)};
    const std::vector<fde::ComplexLinearizedStateArtifact> energies = {
        linearized("D", 5.0), linearized("A", 4.4)};
    const auto overlap = metric();
    const auto result = fde::DiabaticPostprocessor::evaluate(runtime, states, energies, overlap);
    ASSERT_EQ(result.orthogonalized_pair_couplings.size(), 1);
    const auto coupling = result.orthogonalized_pair_couplings[0].coupling_ry;
    EXPECT_NEAR(coupling.real(), -0.5, 1.0e-12);
    EXPECT_NEAR(coupling.imag(), 0.25, 1.0e-12);
    EXPECT_NEAR(result.adiabatic_solution.eigenvalues_ry[0], 3.75, 1.0e-12);
    EXPECT_NEAR(result.adiabatic_solution.eigenvalues_ry[1], 5.25, 1.0e-12);
    EXPECT_LE(result.adiabatic_solution.maximum_residual, 1.0e-12);
    EXPECT_EQ(result.resource_fingerprint, "fixed_resources");
    EXPECT_EQ(result.boundary_twist, runtime.boundary_twist);
}

TEST(FdeComplexDiabaticPostprocess, RejectsWrongBoundaryAndFragmentPopulation)
{
    auto runtime = config();
    const std::vector<fde::ComplexDiabaticDeterminantArtifact> states = {
        determinant("D", 0), determinant("A", 1)};
    const std::vector<fde::ComplexLinearizedStateArtifact> energies = {
        linearized("D", 5.0), linearized("A", 4.4)};
    const auto overlap = metric();
    runtime.boundary_twist[0] = -0.17;
    EXPECT_THROW(fde::DiabaticPostprocessor::evaluate(runtime, states, energies, overlap), std::invalid_argument);
    runtime = config();
    runtime.fragments[0].neutral_valence_electrons += 2;
    EXPECT_THROW(fde::DiabaticPostprocessor::evaluate(runtime, states, energies, overlap), std::invalid_argument);
}

class FdeComplexPostprocessIO : public testing::Test
{
  protected:
    std::string prefix;
    std::string config_path;
    std::string overlap_path;
    std::vector<std::string> paths;

    void SetUp() override
    {
        const auto* info = testing::UnitTest::GetInstance()->current_test_info();
        prefix = testing::TempDir() + "fde-complex-post-" + info->name();
        config_path = prefix + ".config";
        overlap_path = prefix + ".matrix";
        auto runtime = config();
        runtime.output_prefix = prefix;
        runtime.ao_overlap_path = overlap_path;
        paths = {config_path, overlap_path, prefix + ".fde_diabatic", prefix + ".fde_diabatic.tsv"};
        const auto overlap = metric();
        for (std::size_t site = 0; site < 2; ++site)
        {
            const std::string label = site == 0 ? "D" : "A";
            const double energy = site == 0 ? 5.0 : 4.4;
            const auto state = determinant(label, site);
            const auto linear = linearized(label, energy);
            const std::string determinant_path = prefix + "." + label + ".det";
            const std::string linear_path = prefix + "." + label + ".linear";
            paths.push_back(determinant_path);
            paths.push_back(linear_path);
            runtime.determinant_artifacts.push_back({label, determinant_path});
            runtime.linearized_state_artifacts.push_back({label, linear_path});
            std::ofstream determinant_output(determinant_path.c_str());
            fde::ComplexDeterminantArtifactIO::write(determinant_output, state, overlap, 1.0e-12);
            std::ofstream linear_output(linear_path.c_str());
            fde::ComplexLinearizedStateArtifactIO::write(linear_output, linear, 1.0e-12);
        }
        std::ofstream matrix_output(overlap_path.c_str());
        matrix_output << "FDE_AO_MATRIX 2\nGEOMETRY fixed_cell\nORBITALS fixed_basis\n"
                      << "BOUNDARY_TWIST 0.17 0.11 0\nRESOURCES fixed_resources\n"
                      << "DIMENSION 3\nVALUES 9";
        for (const auto& value : overlap) matrix_output << ' ' << value;
        matrix_output << "\nEND\n";
        std::ofstream config_output(config_path.c_str());
        fde::FdeRuntimeConfigIO::write(config_output, runtime);
    }

    void TearDown() override
    {
        for (const auto& path : paths) std::remove(path.c_str());
    }
};

TEST_F(FdeComplexPostprocessIO, DispatchesV2FilesAndWritesCompleteComplexRoots)
{
    std::ostringstream log;
    const auto roots = fde::DiabaticPostprocessor::run_and_count_roots(config_path, log);
    EXPECT_EQ(roots, 2);
    const std::string path = prefix + ".fde_diabatic";
    std::ifstream input(path.c_str());
    std::string header;
    int version = 0;
    input >> header >> version;
    EXPECT_EQ(header, "FDE_DIABATIC_RESULT");
    EXPECT_EQ(version, 2);
    EXPECT_NE(log.str().find("Wrote file: "), std::string::npos);
}

TEST_F(FdeComplexPostprocessIO, RejectsAoMatrixFromAnotherBoundary)
{
    std::ofstream output(overlap_path.c_str());
    output << "FDE_AO_MATRIX 2\nGEOMETRY fixed_cell\nORBITALS fixed_basis\n"
           << "BOUNDARY_TWIST -0.17 0.11 0\nRESOURCES fixed_resources\n"
           << "DIMENSION 3\nVALUES 9";
    const auto overlap = metric();
    for (const auto& value : overlap) output << ' ' << value;
    output << "\nEND\n";
    output.close();
    std::ostringstream log;
    EXPECT_THROW(fde::DiabaticPostprocessor::run_and_count_roots(config_path, log), std::invalid_argument);
}
