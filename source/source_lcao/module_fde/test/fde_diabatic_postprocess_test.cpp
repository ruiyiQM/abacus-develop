#include "../coupling/fde_diabatic_postprocess.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <csignal>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace
{

fde::RuntimeStateDefinition runtime_state(const std::string& label,
                                          const int f_charge,
                                          const int f_spin)
{
    fde::RuntimeStateDefinition result;
    result.state.label = label;
    result.total_charge = -1;
    result.total_spin_projection = 0;
    result.state.fragments.push_back({"F", f_charge, f_spin});
    result.state.fragments.push_back({"CH3Cl", -1 - f_charge, -f_spin});
    return result;
}

fde::FdeRuntimeConfig config()
{
    fde::FdeRuntimeConfig result;
    result.atom_count = 2;
    result.fragments.push_back({"F", {0}, 7});
    result.fragments.push_back({"CH3Cl", {1}, 14});
    result.states.push_back(runtime_state("reactant", -1, 0));
    result.states.push_back(runtime_state("product", 0, 1));
    result.diagonal_energies.push_back({"reactant", -10.0});
    result.diagonal_energies.push_back({"product", -9.0});
    result.output_prefix = "result";
    result.update_order = {"F", "CH3Cl"};
    result.k_state_labels = {"reactant", "product"};
    result.l_fragment_labels = {"F", "CH3Cl"};
    result.m_fragment_labels = {"F", "CH3Cl"};
    return result;
}

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

fde::LinearizedStateArtifact linearized(const std::string& label,
                                        const double energy,
                                        const double off_diagonal)
{
    fde::LinearizedStateArtifact result;
    result.schema_version = 1;
    result.state_label = label;
    result.geometry_fingerprint = "geometry";
    result.ao_dimension = 2;
    result.reference_energy_ry = energy;
    result.hamiltonian_alpha_ry = {2.0, off_diagonal, off_diagonal, 1.0};
    result.hamiltonian_beta_ry = {1.0, 0.0, 0.0, 3.0};
    return result;
}

} // namespace

TEST(FdeDiabaticPostprocess, ProducesCouplingAndNonorthogonalRoots)
{
    const fde::DiabaticPostprocessResult result
        = fde::DiabaticPostprocessor::evaluate(
            config(),
            {determinant("reactant", {1.0, 0.0}),
             determinant("product", {0.8, 0.6})},
            {linearized("reactant", -10.0, 0.7),
             linearized("product", -9.0, 0.3)},
            {1.0, 0.0, 0.0, 1.0});

    ASSERT_EQ(result.assembly.pairs.size(), 1);
    EXPECT_NEAR(result.assembly.pairs[0].normalized_overlap, 0.8, 1.0e-12);
    EXPECT_EQ(result.assembly.pairs[0].provider, "symmetric_linearized");
    EXPECT_LE(result.assembly.pairs[0].overlap_reciprocity_error, 1.0e-12);
    EXPECT_LE(result.assembly.pairs[0].maximum_transition_density_trace_error,
              1.0e-12);
    EXPECT_TRUE(std::isfinite(
        result.assembly.pairs[0].estimated_coupling_uncertainty_ry));
    ASSERT_EQ(result.orthogonalized_pair_couplings.size(), 1);
    EXPECT_TRUE(std::isfinite(result.orthogonalized_pair_couplings[0].coupling_ry));
    ASSERT_EQ(result.adiabatic_solution.eigenvalues_ry.size(), 2);
    EXPECT_LT(result.adiabatic_solution.eigenvalues_ry[0],
              result.adiabatic_solution.eigenvalues_ry[1]);
}

#if defined(__unix__) || defined(__APPLE__)
class FdeDiabaticPostprocessIO : public testing::Test
{
  protected:
    std::string prefix;
    std::string config_path;
    std::vector<std::string> paths;

    void SetUp() override
    {
        const int pid = getpid();
        const std::string pid_text = std::to_string(pid);
        const testing::TestInfo* info = testing::UnitTest::GetInstance()->current_test_info();
        prefix = testing::TempDir() + "fde-postprocess-" + info->name() + "-" + pid_text;
        config_path = prefix + ".config";
        fde::FdeRuntimeConfig runtime = config();
        runtime.output_prefix = prefix;
        runtime.ao_overlap_path = prefix + ".overlap";
        paths = {config_path, runtime.ao_overlap_path,
                 prefix + ".fde_diabatic", prefix + ".fde_diabatic.tsv"};
        const std::vector<double> overlap = {1.0, 0.0, 0.0, 1.0};
        const std::vector<fde::DiabaticDeterminantArtifact> determinants = {
            determinant("reactant", {1.0, 0.0}), determinant("product", {0.8, 0.6})};
        const std::vector<fde::LinearizedStateArtifact> states = {
            linearized("reactant", -10.0, 0.7), linearized("product", -9.0, 0.3)};
        for (std::size_t index = 0; index < states.size(); ++index)
        {
            const std::string determinant_path = prefix + "." + states[index].state_label + ".det";
            const std::string linear_path = prefix + "." + states[index].state_label + ".linear";
            paths.push_back(determinant_path);
            paths.push_back(linear_path);
            runtime.determinant_artifacts.push_back({states[index].state_label, determinant_path});
            runtime.linearized_state_artifacts.push_back({states[index].state_label, linear_path});
            std::ofstream determinant_output(determinant_path.c_str());
            fde::DeterminantArtifactIO::write(determinant_output, determinants[index], overlap, 1.0e-8);
            std::ofstream linear_output(linear_path.c_str());
            fde::LinearizedStateArtifactIO::write(linear_output, states[index], 1.0e-10);
        }
        std::ofstream overlap_output(runtime.ao_overlap_path.c_str());
        overlap_output << "FDE_AO_MATRIX 1\nDIMENSION 2\nVALUES 4 1 0 0 1\nEND\n";
        std::ofstream config_output(config_path.c_str());
        fde::FdeRuntimeConfigIO::write(config_output, runtime);
    }

    void TearDown() override
    {
        for (const std::string& path : paths)
        {
            std::remove(path.c_str());
        }
    }
};

TEST_F(FdeDiabaticPostprocessIO, WritesCompleteResultAndTable)
{
    EXPECT_NO_THROW(fde::DiabaticPostprocessor::run(config_path));
    const std::string text_path = prefix + ".fde_diabatic";
    const std::string table_path = prefix + ".fde_diabatic.tsv";
    std::ifstream text(text_path.c_str());
    std::ifstream table(table_path.c_str());
    const int first_text_character = text.peek();
    const int first_table_character = table.peek();
    const int end_of_file = std::ifstream::traits_type::eof();
    EXPECT_NE(first_text_character, end_of_file);
    EXPECT_NE(first_table_character, end_of_file);
}

TEST_F(FdeDiabaticPostprocessIO, RejectsTextCloseFailure)
{
    EXPECT_EXIT({
        std::signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit;
        limit.rlim_cur = 0;
        limit.rlim_max = 0;
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(2);
        try { fde::DiabaticPostprocessor::run(config_path); }
        catch (const std::runtime_error&) { _exit(0); }
        _exit(1);
    }, testing::ExitedWithCode(0), "");
}

TEST_F(FdeDiabaticPostprocessIO, RejectsTableCloseFailure)
{
    const std::string text_path = prefix + ".fde_diabatic";
    const int link_status = symlink("/dev/null", text_path.c_str());
    ASSERT_EQ(link_status, 0);
    EXPECT_EXIT({
        std::signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit;
        limit.rlim_cur = 0;
        limit.rlim_max = 0;
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(2);
        try { fde::DiabaticPostprocessor::run(config_path); }
        catch (const std::runtime_error&) { _exit(0); }
        _exit(1);
    }, testing::ExitedWithCode(0), "");
}
#endif
