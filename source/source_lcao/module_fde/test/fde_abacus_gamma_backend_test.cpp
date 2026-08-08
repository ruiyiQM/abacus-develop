#include "../fde_abacus_gamma_backend.h"

#include "source_hamilt/module_hcontainer/atom_pair.h"
#include "source_hamilt/module_hcontainer/hcontainer.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace
{

class RecordingGridIntegrator : public fde::GammaGridIntegrator
{
  public:
    void potential_to_ao(const double* potential_ry,
                         hamilt::HContainer<double>& matrix) const override
    {
        for (std::size_t index = 0; index < matrix.size_atom_pairs(); ++index)
        {
            hamilt::AtomPair<double>& pair = matrix.get_atom_pair(index);
            const bool diagonal = pair.get_begin_row() == pair.get_begin_col();
            pair.get_HR_values(0).get_value(0, 0)
                = diagonal ? potential_ry[0] : potential_ry[1];
        }
    }

    void density_to_grid(const std::vector<hamilt::HContainer<double>*>& density_matrices,
                         const int spin_channels,
                         double** density_bohr3) const override
    {
        ASSERT_EQ(spin_channels, 2);
        ASSERT_EQ(density_matrices.size(), 2);
        for (int spin = 0; spin < spin_channels; ++spin)
        {
            density_bohr3[spin][0]
                = density_matrices[spin]->get_atom_pair(0, 0).get_HR_values(0).get_value(0, 0);
            density_bohr3[spin][1]
                = density_matrices[spin]->get_atom_pair(1, 1).get_HR_values(0).get_value(0, 0);
        }
    }

    std::vector<double> local_potential_force(
        const std::vector<const double*>& potential_ry,
        const std::vector<hamilt::HContainer<double>*>& density_matrices,
        const int spin_channels,
        const std::size_t atom_count) const override
    {
        EXPECT_EQ(spin_channels, 2);
        EXPECT_EQ(atom_count, 2);
        std::vector<double> force(3 * atom_count, 0.0);
        for (int spin = 0; spin < spin_channels; ++spin)
        {
            force[0]
                += potential_ry[spin][0]
                   * density_matrices[spin]->get_atom_pair(0, 0)
                         .get_HR_values(0).get_value(0, 0);
            force[3]
                += potential_ry[spin][1]
                   * density_matrices[spin]->get_atom_pair(1, 1)
                         .get_HR_values(0).get_value(0, 0);
        }
        return force;
    }
};

hamilt::HContainer<double> gamma_structure()
{
    const int atom_begin[3] = {0, 1, 2};
    hamilt::HContainer<double> structure(2);
    for (int first = 0; first < 2; ++first)
    {
        for (int second = 0; second < 2; ++second)
        {
            structure.insert_pair(
                hamilt::AtomPair<double>(first, second, atom_begin, atom_begin, 2));
        }
    }
    structure.allocate(nullptr, true);
    structure.fix_gamma();
    return structure;
}

} // namespace

TEST(FdeAbacusGammaBackend, ConvertsPotentialAndDensityThroughGammaHContainer)
{
    hamilt::HContainer<double> structure = gamma_structure();
    const std::shared_ptr<const fde::GammaGridIntegrator> integrator(
        new RecordingGridIntegrator);
    const fde::AbacusGammaBackend backend(structure, 2, integrator);

    const fde::SpinAoMatrix potential
        = backend.embedding_potential_matrix({{1.0, 2.0}, {3.0, 4.0}}, 2);
    EXPECT_EQ(potential.alpha, (std::vector<double>{1.0, 2.0, 2.0, 1.0}));
    EXPECT_EQ(potential.beta, (std::vector<double>{3.0, 4.0, 4.0, 3.0}));

    const fde::SpinDensity density
        = backend.density_from_ao_matrices({{0.8, 0.1, 0.1, 0.2},
                                            {0.6, 0.0, 0.0, 0.4}},
                                           2,
                                           {2, 1, 1, 1.0, 1.0, 1.0});
    EXPECT_EQ(density.alpha_bohr3, (std::vector<double>{0.8, 0.2}));
    EXPECT_EQ(density.beta_bohr3, (std::vector<double>{0.6, 0.4}));
}

TEST(FdeAbacusGammaBackend, RejectsMismatchedDenseAoContract)
{
    hamilt::HContainer<double> structure = gamma_structure();
    const std::shared_ptr<const fde::GammaGridIntegrator> integrator(
        new RecordingGridIntegrator);
    const fde::AbacusGammaBackend backend(structure, 2, integrator);

    EXPECT_THROW(backend.embedding_potential_matrix({{1.0, 2.0}, {3.0, 4.0}}, 3),
                 std::invalid_argument);
}

TEST(FdeAbacusGammaBackend, ContractsLocalPotentialForceThroughGammaHContainer)
{
    hamilt::HContainer<double> structure = gamma_structure();
    const std::shared_ptr<const fde::GammaGridIntegrator> integrator(
        new RecordingGridIntegrator);
    const fde::AbacusGammaBackend backend(structure, 2, integrator);

    const std::vector<double> force
        = backend.force_from_local_potential(
            {{2.0, 3.0}, {5.0, 7.0}},
            {{0.8, 0.1, 0.1, 0.2}, {0.6, 0.0, 0.0, 0.4}},
            2,
            2);

    ASSERT_EQ(force.size(), 6);
    EXPECT_NEAR(force[0], 4.6, 1.0e-14);
    EXPECT_DOUBLE_EQ(force[1], 0.0);
    EXPECT_DOUBLE_EQ(force[2], 0.0);
    EXPECT_NEAR(force[3], 3.4, 1.0e-14);
    EXPECT_DOUBLE_EQ(force[4], 0.0);
    EXPECT_DOUBLE_EQ(force[5], 0.0);
}
