#include "../fde_diabatic_postprocess.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

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
    ASSERT_EQ(result.orthogonalized_pair_couplings.size(), 1);
    EXPECT_TRUE(std::isfinite(result.orthogonalized_pair_couplings[0].coupling_ry));
    ASSERT_EQ(result.adiabatic_solution.eigenvalues_ry.size(), 2);
    EXPECT_LT(result.adiabatic_solution.eigenvalues_ry[0],
              result.adiabatic_solution.eigenvalues_ry[1]);
}
