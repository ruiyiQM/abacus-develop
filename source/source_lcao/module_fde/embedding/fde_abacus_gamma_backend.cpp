#include "source_lcao/module_fde/embedding/fde_abacus_gamma_backend.h"

#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_hamilt/module_hcontainer/atom_pair.h"
#include "source_hamilt/module_hcontainer/hcontainer.h"

#include <algorithm>
#include <stdexcept>

namespace fde
{

namespace
{

std::size_t structure_dimension(const hamilt::HContainer<double>& structure)
{
    std::size_t dimension = 0;
    for (std::size_t pair_index = 0; pair_index < structure.size_atom_pairs(); ++pair_index)
    {
        const hamilt::AtomPair<double>& pair = structure.get_atom_pair(pair_index);
        dimension = std::max(dimension,
                             static_cast<std::size_t>(pair.get_begin_row() + pair.get_row_size()));
        dimension = std::max(dimension,
                             static_cast<std::size_t>(pair.get_begin_col() + pair.get_col_size()));
    }
    return dimension;
}

std::vector<double> container_to_dense(const hamilt::HContainer<double>& matrix,
                                       const std::size_t dimension)
{
    std::vector<double> dense(dimension * dimension, 0.0);
    for (std::size_t pair_index = 0; pair_index < matrix.size_atom_pairs(); ++pair_index)
    {
        const hamilt::AtomPair<double>& pair = matrix.get_atom_pair(pair_index);
        if (pair.get_R_size() != 1)
        {
            throw std::invalid_argument("FDE Gamma backend requires one merged R block per atom pair");
        }
        pair.add_to_matrix(0, dense.data(), static_cast<int>(dimension), 1.0, 0);
    }
    return dense;
}

void dense_to_container(const std::vector<double>& dense,
                        const std::size_t dimension,
                        hamilt::HContainer<double>& matrix)
{
    matrix.set_zero();
    for (std::size_t pair_index = 0; pair_index < matrix.size_atom_pairs(); ++pair_index)
    {
        hamilt::AtomPair<double>& pair = matrix.get_atom_pair(pair_index);
        if (pair.get_R_size() != 1)
        {
            throw std::invalid_argument("FDE Gamma backend requires one merged R block per atom pair");
        }
        pair.add_from_matrix(0, dense.data(), static_cast<int>(dimension), 1.0, 0);
    }
}

} // namespace

AbacusGammaBackend::AbacusGammaBackend(
    const hamilt::HContainer<double>& gamma_structure,
    const std::size_t local_grid_size,
    const std::shared_ptr<const GammaGridIntegrator>& integrator)
    : gamma_structure_(&gamma_structure),
      local_grid_size_(local_grid_size),
      integrator_(integrator)
{
    if (!gamma_structure.is_gamma_only() || local_grid_size == 0 || !integrator)
    {
        throw std::invalid_argument(
            "FDE Gamma backend requires a Gamma-fixed AO structure, a grid, and an integrator");
    }
}

void AbacusGammaBackend::validate_ao_contract(const std::size_t full_ao_dimension) const
{
    if (full_ao_dimension == 0 || structure_dimension(*gamma_structure_) != full_ao_dimension)
    {
        throw std::invalid_argument("FDE dense AO dimension does not match the HContainer structure");
    }
    const Parallel_Orbitals* orbitals = gamma_structure_->get_paraV();
    if (orbitals != nullptr
        && (orbitals->get_row_size() != orbitals->get_global_row_size()
            || orbitals->get_col_size() != orbitals->get_global_col_size()))
    {
        throw std::invalid_argument(
            "FDE dense Gamma backend does not accept a distributed 2D-block AO matrix");
    }
}

SpinAoMatrix AbacusGammaBackend::embedding_potential_matrix(
    const SpinPotential& real_space_potential,
    const std::size_t full_ao_dimension) const
{
    this->validate_ao_contract(full_ao_dimension);
    if (real_space_potential.alpha_ry.size() != local_grid_size_
        || real_space_potential.beta_ry.size() != local_grid_size_)
    {
        throw std::invalid_argument("FDE embedding potential does not match the local ABACUS grid");
    }

    hamilt::HContainer<double> alpha(*gamma_structure_);
    integrator_->potential_to_ao(real_space_potential.alpha_ry.data(), alpha);
    hamilt::HContainer<double> beta(*gamma_structure_);
    integrator_->potential_to_ao(real_space_potential.beta_ry.data(), beta);

    SpinAoMatrix result;
    result.alpha = container_to_dense(alpha, full_ao_dimension);
    result.beta = container_to_dense(beta, full_ao_dimension);
    return result;
}

SpinDensity AbacusGammaBackend::density_from_ao_matrices(
    const SpinAoMatrix& density_matrices,
    const std::size_t full_ao_dimension,
    const UniformGrid& grid) const
{
    this->validate_ao_contract(full_ao_dimension);
    const std::size_t matrix_size = full_ao_dimension * full_ao_dimension;
    if (density_matrices.alpha.size() != matrix_size
        || density_matrices.beta.size() != matrix_size
        || grid.x * grid.y * grid.z != local_grid_size_)
    {
        throw std::invalid_argument("FDE density matrices or grid do not match the ABACUS backend");
    }

    hamilt::HContainer<double> alpha(*gamma_structure_);
    hamilt::HContainer<double> beta(*gamma_structure_);
    dense_to_container(density_matrices.alpha, full_ao_dimension, alpha);
    dense_to_container(density_matrices.beta, full_ao_dimension, beta);

    SpinDensity result;
    result.alpha_bohr3.assign(local_grid_size_, 0.0);
    result.beta_bohr3.assign(local_grid_size_, 0.0);
    double* density[2] = {result.alpha_bohr3.data(), result.beta_bohr3.data()};
    std::vector<hamilt::HContainer<double>*> matrices;
    matrices.push_back(&alpha);
    matrices.push_back(&beta);
    integrator_->density_to_grid(matrices, 2, density);
    return result;
}

std::vector<double> AbacusGammaBackend::force_from_local_potential(
    const SpinPotential& potential,
    const SpinAoMatrix& density_matrix,
    const std::size_t full_ao_dimension,
    const std::size_t atom_count) const
{
    this->validate_ao_contract(full_ao_dimension);
    const std::size_t matrix_size = full_ao_dimension * full_ao_dimension;
    if (atom_count == 0 || potential.alpha_ry.size() != local_grid_size_
        || potential.beta_ry.size() != local_grid_size_
        || density_matrix.alpha.size() != matrix_size
        || density_matrix.beta.size() != matrix_size)
    {
        throw std::invalid_argument(
            "FDE local-potential force inputs do not match the ABACUS backend");
    }

    hamilt::HContainer<double> alpha(*gamma_structure_);
    hamilt::HContainer<double> beta(*gamma_structure_);
    dense_to_container(density_matrix.alpha, full_ao_dimension, alpha);
    dense_to_container(density_matrix.beta, full_ao_dimension, beta);
    std::vector<hamilt::HContainer<double>*> matrices;
    matrices.push_back(&alpha);
    matrices.push_back(&beta);
    std::vector<const double*> potentials;
    potentials.push_back(potential.alpha_ry.data());
    potentials.push_back(potential.beta_ry.data());
    const std::vector<double> force
        = integrator_->local_potential_force(potentials, matrices, 2, atom_count);
    if (force.size() != 3 * atom_count)
    {
        throw std::runtime_error(
            "FDE Gint local-potential force returned an invalid Cartesian array");
    }
    return force;
}

} // namespace fde
