#include "../fde_one_way_scf.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{

class TargetDensityBackend : public fde::OneWayScfBackend
{
  public:
    explicit TargetDensityBackend(const fde::SpinDensity& target) : target_(target) {}

    fde::SpinAoMatrix embedding_potential_matrix(
        const fde::SpinPotential& real_space_potential,
        const std::size_t full_ao_dimension) const override
    {
        if (real_space_potential.alpha_ry.size() != 2)
        {
            throw std::invalid_argument("unexpected test potential size");
        }
        return {std::vector<double>(full_ao_dimension * full_ao_dimension, 0.0),
                std::vector<double>(full_ao_dimension * full_ao_dimension, 0.0)};
    }

    fde::SpinDensity density_from_ao_matrices(
        const fde::SpinAoMatrix& density_matrices,
        const std::size_t full_ao_dimension,
        const fde::UniformGrid& grid) const override
    {
        if (density_matrices.alpha.size() != full_ao_dimension * full_ao_dimension
            || grid.x * grid.y * grid.z != 2)
        {
            throw std::invalid_argument("unexpected test density matrix size");
        }
        return target_;
    }

  private:
    fde::SpinDensity target_;
};

fde::FrozenDensityArtifact artifact(const std::string& fragment,
                                    const std::vector<double>& alpha,
                                    const std::vector<double>& beta)
{
    fde::FrozenDensityArtifact value;
    value.schema_version = 1;
    value.fragment_label = fragment;
    value.state_label = "state_1";
    value.geometry_fingerprint = "geometry";
    value.grid_fingerprint = "grid";
    value.pseudopotential_fingerprint = "pseudo";
    value.orbital_fingerprint = "orbital";
    value.core_density_fingerprint = fragment + "_core";
    value.xc_functional = "PBE";
    value.kinetic_functional = "PW91K";
    value.grid_x = 2;
    value.grid_y = 1;
    value.grid_z = 1;
    value.cell_volume_bohr3 = 2.0;
    value.alpha_electrons = 1;
    value.beta_electrons = 1;
    value.freeze_thaw_cycle = 0;
    value.scf_converged = true;
    value.orbital_kinetic_energy_ry = 0.0;
    value.nonlocal_pseudopotential_energy_ry = 0.0;
    value.rho_alpha_bohr3 = alpha;
    value.rho_beta_bohr3 = beta;
    return value;
}

fde::OneWayScfRequest request()
{
    fde::OneWayScfRequest value;
    value.active_initial = artifact("A", {0.8, 0.2}, {0.6, 0.4});
    value.frozen = artifact("B", {0.3, 0.7}, {0.4, 0.6});
    value.frozen_hartree_potential_ry = {0.0, 0.0};
    value.full_hamiltonian_ry = {-1.0, 0.0, 0.0, -0.5};
    value.full_overlap = {1.0, 0.0, 0.0, 1.0};
    value.full_ao_dimension = 2;
    value.active_orbitals = {0, 1};
    value.potential_config = {{2, 1, 1, 1.0, 1.0, 1.0},
                              fde::KineticFunctional::ThomasFermi,
                              1.0e-12};
    value.controls = {5, 1.0e-12, 1.0e-10, 1.0};
    return value;
}

} // namespace

TEST(FdeOneWayScf, ConvergesAndExportsUpdatedArtifact)
{
    const fde::SpinDensity target{{0.5, 0.5}, {0.5, 0.5}};
    const TargetDensityBackend backend(target);
    const fde::DiracExchangeProvider xc;

    const fde::OneWayScfResult result = fde::OneWayScf::run(request(), xc, backend);

    EXPECT_TRUE(result.active.scf_converged);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_EQ(result.active.freeze_thaw_cycle, 1);
    EXPECT_EQ(result.active.rho_alpha_bohr3, target.alpha_bohr3);
    EXPECT_NEAR(result.density_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(fde::SubspaceSolver::electron_count(result.alpha_solution.active_density_matrix,
                                                    request().full_overlap,
                                                    2),
                1.0,
                1.0e-12);
}

TEST(FdeOneWayScf, RejectsIncompatibleFrozenArtifact)
{
    fde::OneWayScfRequest invalid = request();
    invalid.frozen.geometry_fingerprint = "other_geometry";
    const TargetDensityBackend backend({{0.5, 0.5}, {0.5, 0.5}});
    const fde::DiracExchangeProvider xc;

    EXPECT_THROW(fde::OneWayScf::run(invalid, xc, backend), std::invalid_argument);
}
