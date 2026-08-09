#include "fde_projected_hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fde
{

FdeProjectedHamiltonian::FdeProjectedHamiltonian(
    hamilt::Hamilt<double>& full_hamiltonian,
    const Parallel_2D& orbital_distribution,
    const std::size_t full_dimension,
    const std::vector<std::size_t>& active_orbitals,
    const double inactive_energy_ry)
    : full_hamiltonian_(&full_hamiltonian),
      orbital_distribution_(&orbital_distribution),
      full_dimension_(full_dimension),
      active_orbitals_(active_orbitals),
      is_active_(full_dimension, false),
      inactive_energy_ry_(inactive_energy_ry),
      projected_hamiltonian_(static_cast<std::size_t>(
                                 orbital_distribution.get_local_size()),
                             0.0),
      projected_overlap_(static_cast<std::size_t>(
                             orbital_distribution.get_local_size()),
                         0.0)
{
    if (full_dimension == 0 || active_orbitals.empty()
        || active_orbitals.size() >= full_dimension
        || orbital_distribution.get_global_row_size()
               != static_cast<int>(full_dimension)
        || orbital_distribution.get_global_col_size()
               != static_cast<int>(full_dimension)
        || orbital_distribution.get_row_size() < 0
        || orbital_distribution.get_col_size() < 0
        || orbital_distribution.get_local_size() < 0
        || !std::isfinite(inactive_energy_ry) || inactive_energy_ry <= 0.0)
    {
        throw std::invalid_argument("FDE projected Hamiltonian dimensions or inactive energy are invalid");
    }
    if (!std::is_sorted(active_orbitals_.begin(), active_orbitals_.end())
        || std::adjacent_find(active_orbitals_.begin(), active_orbitals_.end())
               != active_orbitals_.end()
        || active_orbitals_.back() >= full_dimension_)
    {
        throw std::invalid_argument("FDE active AO indices must be sorted, unique, and in range");
    }
    for (std::size_t index = 0; index < active_orbitals_.size(); ++index)
    {
        is_active_[active_orbitals_[index]] = true;
    }
    this->classname = "FdeProjectedHamiltonian";
}

void FdeProjectedHamiltonian::updateHk(const int kpoint)
{
    full_hamiltonian_->updateHk(kpoint);
}

void FdeProjectedHamiltonian::refresh(const bool yes)
{
    full_hamiltonian_->refresh(yes);
}

void FdeProjectedHamiltonian::matrix(hamilt::MatrixBlock<double>& hamiltonian,
                                     hamilt::MatrixBlock<double>& overlap)
{
    hamilt::MatrixBlock<double> full_h;
    hamilt::MatrixBlock<double> full_s;
    full_hamiltonian_->matrix(full_h, full_s);
    const std::size_t local_rows
        = static_cast<std::size_t>(orbital_distribution_->get_row_size());
    const std::size_t local_columns
        = static_cast<std::size_t>(orbital_distribution_->get_col_size());
    if (full_h.p == nullptr || full_s.p == nullptr || full_h.row != local_rows
        || full_h.col != local_columns || full_s.row != local_rows
        || full_s.col != local_columns)
    {
        throw std::invalid_argument(
            "FDE projected Hamiltonian matrices do not match Parallel_Orbitals");
    }

    std::fill(projected_hamiltonian_.begin(), projected_hamiltonian_.end(), 0.0);
    std::fill(projected_overlap_.begin(), projected_overlap_.end(), 0.0);
    for (std::size_t local_column = 0; local_column < local_columns; ++local_column)
    {
        const std::size_t global_column = static_cast<std::size_t>(
            orbital_distribution_->local2global_col(static_cast<int>(local_column)));
        for (std::size_t local_row = 0; local_row < local_rows; ++local_row)
        {
            const std::size_t global_row = static_cast<std::size_t>(
                orbital_distribution_->local2global_row(static_cast<int>(local_row)));
            const std::size_t offset = local_row + local_column * local_rows;
            if (is_active_[global_row] && is_active_[global_column])
            {
                projected_hamiltonian_[offset] = full_h.p[offset];
                projected_overlap_[offset] = full_s.p[offset];
            }
            else if (global_row == global_column)
            {
                projected_hamiltonian_[offset] = inactive_energy_ry_;
                projected_overlap_[offset] = 1.0;
            }
        }
    }
    hamiltonian = hamilt::MatrixBlock<double>{projected_hamiltonian_.data(),
                                              local_rows,
                                              local_columns,
                                              full_h.desc};
    overlap = hamilt::MatrixBlock<double>{projected_overlap_.data(),
                                          local_rows,
                                          local_columns,
                                          full_s.desc};
}

const std::vector<std::size_t>& FdeProjectedHamiltonian::active_orbitals() const
{
    return active_orbitals_;
}

} // namespace fde
