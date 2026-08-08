#include "../fde_determinant_artifact.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

fde::ActiveSubspaceSolution solution(const std::vector<double>& occupations,
                                     const double energy_shift)
{
    fde::ActiveSubspaceSolution value;
    value.eigenvalues_ry = {-1.0 + energy_shift, -0.5 + energy_shift};
    value.eigenvectors = {1.0, 0.0, 0.0, 1.0};
    value.occupations = occupations;
    return value;
}

fde::FragmentOrbitalSolution fragment(const std::string& label,
                                      const std::vector<std::size_t>& active_orbitals,
                                      const int alpha_electrons,
                                      const int beta_electrons,
                                      const double energy_shift)
{
    fde::FragmentOrbitalSolution value;
    value.fragment_label = label;
    value.active_orbitals = active_orbitals;
    value.alpha_electrons = alpha_electrons;
    value.beta_electrons = beta_electrons;
    value.alpha_solution = solution(alpha_electrons == 0
                                        ? std::vector<double>{0.0, 0.0}
                                        : std::vector<double>{1.0, 0.0},
                                    energy_shift);
    value.beta_solution = solution(beta_electrons == 0
                                       ? std::vector<double>{0.0, 0.0}
                                       : std::vector<double>{1.0, 0.0},
                                   energy_shift);
    return value;
}

std::vector<double> ao_overlap()
{
    std::vector<double> overlap(16, 0.0);
    for (std::size_t index = 0; index < 4; ++index)
    {
        overlap[index + index * 4] = 1.0;
    }
    overlap[0 + 2 * 4] = 0.2;
    overlap[2 + 0 * 4] = 0.2;
    return overlap;
}

} // namespace

TEST(FdeDeterminantArtifact, BuildsFullAoOccupiedColumnsAndPreservesCrossFragmentOverlap)
{
    const std::vector<fde::FragmentOrbitalSolution> fragments
        = {fragment("A", {0, 1}, 1, 1, 0.0),
           fragment("B", {2, 3}, 1, 0, 0.1)};
    const fde::DiabaticDeterminantArtifact artifact
        = fde::DeterminantArtifactIO::build({"charge_A", "geometry", "basis"},
                                            4,
                                            fragments,
                                            ao_overlap(),
                                            1.0e-12);

    EXPECT_EQ(fde::DeterminantArtifactIO::occupied_count(artifact.alpha, 4), 2);
    EXPECT_EQ(fde::DeterminantArtifactIO::occupied_count(artifact.beta, 4), 1);
    EXPECT_EQ(artifact.alpha.coefficients,
              (std::vector<double>{1.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0, 0.0}));
    EXPECT_EQ(artifact.alpha.source_fragment_labels,
              (std::vector<std::string>{"A", "B"}));

    std::ostringstream output;
    fde::DeterminantArtifactIO::write(output, artifact, ao_overlap(), 1.0e-12);
    std::istringstream input(output.str());
    const fde::DiabaticDeterminantArtifact restored
        = fde::DeterminantArtifactIO::read(input, ao_overlap(), 1.0e-12);
    EXPECT_EQ(restored.alpha.coefficients, artifact.alpha.coefficients);
    EXPECT_EQ(restored.beta.orbital_energies_ry, artifact.beta.orbital_energies_ry);
}

TEST(FdeDeterminantArtifact, RejectsFractionalOccupations)
{
    std::vector<fde::FragmentOrbitalSolution> fragments
        = {fragment("A", {0, 1}, 1, 1, 0.0),
           fragment("B", {2, 3}, 1, 0, 0.1)};
    fragments[0].alpha_solution.occupations = {0.5, 0.5};

    EXPECT_THROW(fde::DeterminantArtifactIO::build({"charge_A", "geometry", "basis"},
                                                   4,
                                                   fragments,
                                                   ao_overlap(),
                                                   1.0e-12),
                 std::invalid_argument);
}

TEST(FdeDeterminantArtifact, RejectsOverlappingFragmentAoSpaces)
{
    const std::vector<fde::FragmentOrbitalSolution> fragments
        = {fragment("A", {0, 1}, 1, 1, 0.0),
           fragment("B", {1, 2}, 1, 0, 0.1)};

    EXPECT_THROW(fde::DeterminantArtifactIO::build({"charge_A", "geometry", "basis"},
                                                   4,
                                                   fragments,
                                                   ao_overlap(),
                                                   1.0e-12),
                 std::invalid_argument);
}
