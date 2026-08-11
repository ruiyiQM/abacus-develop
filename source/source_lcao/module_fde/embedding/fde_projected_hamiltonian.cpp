#include "source_lcao/module_fde/embedding/fde_projected_hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace fde
{

template <typename TK>
FdeProjectedHamiltonianT<TK>::FdeProjectedHamiltonianT(
    hamilt::Hamilt<TK>& full_hamiltonian,
    const Parallel_2D& orbital_distribution,
    const std::size_t full_dimension,
    const std::vector<std::size_t>& active_orbitals,
    const double inactive_energy_ry,
    const std::size_t kpoint_count,
    const std::vector<std::size_t>& overlap_slots)
    : full_hamiltonian_(&full_hamiltonian),
      orbital_distribution_(&orbital_distribution),
      full_dimension_(full_dimension),
      active_orbitals_(active_orbitals),
      is_active_(full_dimension, false),
      inactive_energy_ry_(inactive_energy_ry),
      kpoint_count_(kpoint_count),
      current_kpoint_(0),
      local_matrix_size_(static_cast<std::size_t>(
          orbital_distribution.get_local_size())),
      overlap_slots_(overlap_slots),
      projected_hamiltonian_(static_cast<std::size_t>(
                                 orbital_distribution.get_local_size()),
                             TK(0.0)),
      projected_overlap_(),
      overlap_initialized_()
{
    if (full_dimension == 0 || active_orbitals.empty() || kpoint_count == 0
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
    if (overlap_slots_.empty())
    {
        overlap_slots_.resize(kpoint_count_);
        for (std::size_t kpoint = 0; kpoint < kpoint_count_; ++kpoint)
        {
            overlap_slots_[kpoint] = kpoint;
        }
    }
    if (overlap_slots_.size() != kpoint_count_)
    {
        throw std::invalid_argument(
            "FDE projected Hamiltonian overlap-slot map has the wrong size");
    }
    const std::size_t overlap_slot_count
        = *std::max_element(overlap_slots_.begin(), overlap_slots_.end()) + 1;
    std::vector<unsigned char> slot_seen(overlap_slot_count, 0);
    for (std::size_t kpoint = 0; kpoint < kpoint_count_; ++kpoint)
    {
        slot_seen[overlap_slots_[kpoint]] = 1;
    }
    if (std::find(slot_seen.begin(), slot_seen.end(), 0) != slot_seen.end())
    {
        throw std::invalid_argument(
            "FDE projected Hamiltonian overlap slots must be contiguous");
    }
    projected_overlap_.assign(local_matrix_size_ * overlap_slot_count, TK(0.0));
    overlap_initialized_.assign(overlap_slot_count, 0);
    this->classname = "FdeProjectedHamiltonian";
}

template <typename TK>
void FdeProjectedHamiltonianT<TK>::updateHk(const int kpoint)
{
    if (kpoint < 0 || static_cast<std::size_t>(kpoint) >= kpoint_count_)
    {
        throw std::out_of_range("FDE projected Hamiltonian k-point index is out of range");
    }
    current_kpoint_ = static_cast<std::size_t>(kpoint);
    full_hamiltonian_->updateHk(kpoint);
}

template <typename TK>
void FdeProjectedHamiltonianT<TK>::refresh(const bool yes)
{
    full_hamiltonian_->refresh(yes);
}

template <typename TK>
void FdeProjectedHamiltonianT<TK>::matrix(
    hamilt::MatrixBlock<TK>& hamiltonian,
    hamilt::MatrixBlock<TK>& overlap)
{
    hamilt::MatrixBlock<TK> full_h;
    hamilt::MatrixBlock<TK> full_s;
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

    std::fill(projected_hamiltonian_.begin(),
              projected_hamiltonian_.end(),
              TK(0.0));
    const std::size_t overlap_slot = overlap_slots_[current_kpoint_];
    const std::size_t overlap_base = overlap_slot * local_matrix_size_;
    if (!overlap_initialized_[overlap_slot])
    {
        std::fill(projected_overlap_.begin() + overlap_base,
                  projected_overlap_.begin() + overlap_base + local_matrix_size_,
                  TK(0.0));
    }
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
                if (!overlap_initialized_[overlap_slot])
                {
                    projected_overlap_[overlap_base + offset] = full_s.p[offset];
                }
            }
            else if (global_row == global_column)
            {
                projected_hamiltonian_[offset] = TK(inactive_energy_ry_);
                if (!overlap_initialized_[overlap_slot])
                {
                    projected_overlap_[overlap_base + offset] = TK(1.0);
                }
            }
        }
    }
    overlap_initialized_[overlap_slot] = 1;
    hamiltonian = hamilt::MatrixBlock<TK>{projected_hamiltonian_.data(),
                                          local_rows,
                                          local_columns,
                                          full_h.desc};
    overlap = hamilt::MatrixBlock<TK>{projected_overlap_.data() + overlap_base,
                                      local_rows,
                                      local_columns,
                                      full_s.desc};
}

template <typename TK>
const std::vector<std::size_t>&
FdeProjectedHamiltonianT<TK>::active_orbitals() const
{
    return active_orbitals_;
}

template class FdeProjectedHamiltonianT<double>;
template class FdeProjectedHamiltonianT<std::complex<double>>;

} // namespace fde
