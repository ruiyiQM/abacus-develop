#include "fde_projected_hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fde
{

FdeProjectedHamiltonian::FdeProjectedHamiltonian(
    hamilt::Hamilt<double>& full_hamiltonian,
    const std::size_t full_dimension,
    const std::vector<std::size_t>& active_orbitals,
    const double inactive_energy_ry)
    : full_hamiltonian_(&full_hamiltonian),
      full_dimension_(full_dimension),
      active_orbitals_(active_orbitals),
      is_active_(full_dimension, false),
      inactive_energy_ry_(inactive_energy_ry),
      projected_hamiltonian_(full_dimension * full_dimension, 0.0),
      projected_overlap_(full_dimension * full_dimension, 0.0)
{
    if (full_dimension == 0 || active_orbitals.empty()
        || active_orbitals.size() >= full_dimension
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
    if (full_h.p == nullptr || full_s.p == nullptr || full_h.row != full_dimension_
        || full_h.col != full_dimension_ || full_s.row != full_dimension_
        || full_s.col != full_dimension_)
    {
        throw std::invalid_argument(
            "FDE projected Hamiltonian requires replicated square Gamma matrices");
    }

    std::fill(projected_hamiltonian_.begin(), projected_hamiltonian_.end(), 0.0);
    std::fill(projected_overlap_.begin(), projected_overlap_.end(), 0.0);
    for (std::size_t column = 0; column < full_dimension_; ++column)
    {
        for (std::size_t row = 0; row < full_dimension_; ++row)
        {
            const std::size_t offset = row + column * full_dimension_;
            if (is_active_[row] && is_active_[column])
            {
                projected_hamiltonian_[offset] = full_h.p[offset];
                projected_overlap_[offset] = full_s.p[offset];
            }
            else if (row == column)
            {
                projected_hamiltonian_[offset] = inactive_energy_ry_;
                projected_overlap_[offset] = 1.0;
            }
        }
    }
    hamiltonian = hamilt::MatrixBlock<double>{projected_hamiltonian_.data(),
                                              full_dimension_,
                                              full_dimension_,
                                              full_h.desc};
    overlap = hamilt::MatrixBlock<double>{projected_overlap_.data(),
                                          full_dimension_,
                                          full_dimension_,
                                          full_s.desc};
}

const std::vector<std::size_t>& FdeProjectedHamiltonian::active_orbitals() const
{
    return active_orbitals_;
}

} // namespace fde
