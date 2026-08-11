#include "../coupling/fde_diabatic_assembler.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

fde::OccupiedSpinOrbitals occupied(
    const std::vector<std::vector<double>>& columns,
    const std::vector<std::string>& fragments)
{
    fde::OccupiedSpinOrbitals result;
    for (std::size_t column = 0; column < columns.size(); ++column)
    {
        result.coefficients.insert(result.coefficients.end(),
                                   columns[column].begin(),
                                   columns[column].end());
        result.orbital_energies_ry.push_back(-1.0 + 0.1 * column);
        result.source_fragment_labels.push_back(fragments[column]);
    }
    return result;
}

fde::DiabaticDeterminantArtifact determinant(
    const std::string& state,
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c)
{
    fde::DiabaticDeterminantArtifact result;
    result.schema_version = 1;
    result.state_label = state;
    result.geometry_fingerprint = "geometry";
    result.orbital_fingerprint = "basis";
    result.ao_dimension = 6;
    result.alpha = occupied({a, b, c}, {"A", "B", "C"});
    return result;
}

std::vector<double> ao_overlap()
{
    std::vector<double> result(36, 0.0);
    for (std::size_t index = 0; index < 6; ++index)
    {
        result[index + index * 6] = 1.0;
    }
    const std::size_t pairs[][2] = {{0, 2}, {0, 3}, {1, 2}, {1, 3},
                                    {0, 4}, {1, 4}, {2, 4}, {3, 4}};
    const double values[] = {0.10, 0.02, 0.03, 0.01, 0.04, 0.02, 0.07, 0.02};
    for (std::size_t index = 0; index < 8; ++index)
    {
        result[pairs[index][0] + pairs[index][1] * 6] = values[index];
        result[pairs[index][1] + pairs[index][0] * 6] = values[index];
    }
    return result;
}

std::vector<fde::DiabaticDeterminantArtifact> determinants()
{
    return {determinant("state_A",
                        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0}),
            determinant("state_B",
                        {0.8, 0.6, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0}),
            determinant("state_C",
                        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.6, 0.8, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0})};
}

class CountingTransitionEnergy : public fde::TransitionEnergyProvider
{
  public:
    std::string name() const override
    {
        return "counting";
    }

    double evaluate_ry(const fde::DiabaticDeterminantArtifact& bra,
                       const fde::DiabaticDeterminantArtifact&,
                       const fde::SpinTransitionDensityMatrix&,
                       const std::size_t) const override
    {
        ++calls;
        return bra.state_label == "state_A" ? 4.0 : 6.0;
    }

    mutable int calls = 0;
};

} // namespace

TEST(FdeDiabaticAssembler, AppliesExplicitKLMSelectionsAndBuildsSymmetricMatrices)
{
    const CountingTransitionEnergy energy;
    const fde::FdeDiabApproximationSpec approximation{{0, 2}, {"A"}, {"A", "B", "C"}};
    const fde::CouplingValidationControls validation{1.0e-12, 1.0e-12};
    const fde::FdeDiabaticAssemblyResult result
        = fde::FdeDiabaticAssembler::assemble(determinants(),
                                              {1.0, 2.0, 3.0},
                                              ao_overlap(),
                                              approximation,
                                              energy,
                                              validation,
                                              1.0e-12);

    EXPECT_EQ(result.problem.state_labels,
              (std::vector<std::string>{"state_A", "state_C"}));
    EXPECT_DOUBLE_EQ(result.problem.hamiltonian_ry[0], 1.0);
    EXPECT_DOUBLE_EQ(result.problem.hamiltonian_ry[3], 3.0);
    EXPECT_DOUBLE_EQ(result.problem.overlap[0], 1.0);
    EXPECT_DOUBLE_EQ(result.problem.overlap[3], 1.0);
    EXPECT_DOUBLE_EQ(result.problem.overlap[1], result.problem.overlap[2]);
    EXPECT_DOUBLE_EQ(result.problem.hamiltonian_ry[1],
                     result.problem.hamiltonian_ry[2]);
    EXPECT_EQ(energy.calls, 2);
    EXPECT_EQ(result.audit.k_states, 2);
    EXPECT_EQ(result.audit.l_overlap_active_fragments, 1);
    EXPECT_EQ(result.audit.m_determinant_fragments, 3);
    EXPECT_EQ(result.audit.omitted_state_indices,
              (std::vector<std::size_t>{1}));
    EXPECT_EQ(result.audit.retained_interfragment_overlap_pairs, 2);
    EXPECT_EQ(result.audit.discarded_interfragment_overlap_pairs, 1);
    ASSERT_EQ(result.pairs.size(), 1);
    EXPECT_EQ(result.pairs[0].first_input_state, 0);
    EXPECT_EQ(result.pairs[0].second_input_state, 2);
    EXPECT_EQ(result.pairs[0].provider, "counting");
    EXPECT_LE(result.pairs[0].maximum_transition_density_trace_error,
              1.0e-12);
}

TEST(FdeDiabaticAssembler, ImplementsPaperLBlockRetentionRule)
{
    const fde::FdeDiabFragmentOverlapPolicy policy({"A"});
    EXPECT_TRUE(policy.retain("A", "B"));
    EXPECT_TRUE(policy.retain("C", "A"));
    EXPECT_TRUE(policy.retain("B", "B"));
    EXPECT_FALSE(policy.retain("B", "C"));
}

TEST(FdeDiabaticAssembler, RejectsLFragmentsOutsideMSelection)
{
    const CountingTransitionEnergy energy;
    const fde::FdeDiabApproximationSpec invalid{{0, 1}, {"C"}, {"A", "B"}};
    const fde::CouplingValidationControls validation{1.0e-12, 1.0e-12};
    EXPECT_THROW(fde::FdeDiabaticAssembler::assemble(determinants(),
                                                     {1.0, 2.0, 3.0},
                                                     ao_overlap(),
                                                     invalid,
                                                     energy,
                                                     validation,
                                                     1.0e-12),
                 std::invalid_argument);
}
