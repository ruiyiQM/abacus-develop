#ifndef FDE_EMBEDDING_PROJECTED_HAMILTONIAN_H
#define FDE_EMBEDDING_PROJECTED_HAMILTONIAN_H

#include "source_base/parallel_2d.h"
#include "source_hamilt/hamilt.h"

#include <cstddef>
#include <complex>
#include <vector>

namespace fde
{

/**
 * Distributed LCAO Hamiltonian view whose low-energy variational space is
 * exactly the selected fragment AO principal subspace.
 *
 * The real specialization serves Gamma-only calculations.  The complex
 * specialization serves general k points and keeps one projected overlap
 * buffer per spin-k index because S(k) is both k dependent and factorized in
 * place by some generalized eigensolvers.
 */
template <typename TK>
class FdeProjectedHamiltonianT : public hamilt::Hamilt<TK>
{
  public:
    FdeProjectedHamiltonianT(hamilt::Hamilt<TK>& full_hamiltonian,
                             const Parallel_2D& orbital_distribution,
                             std::size_t full_dimension,
                             const std::vector<std::size_t>& active_orbitals,
                             double inactive_energy_ry,
                             std::size_t kpoint_count = 1,
                             const std::vector<std::size_t>& overlap_slots
                                 = std::vector<std::size_t>());

    void updateHk(int kpoint) override;
    void refresh(bool yes) override;
    void matrix(hamilt::MatrixBlock<TK>& hamiltonian,
                hamilt::MatrixBlock<TK>& overlap) override;

    const std::vector<std::size_t>& active_orbitals() const;

  private:
    hamilt::Hamilt<TK>* full_hamiltonian_;
    const Parallel_2D* orbital_distribution_;
    std::size_t full_dimension_;
    std::vector<std::size_t> active_orbitals_;
    std::vector<bool> is_active_;
    double inactive_energy_ry_;
    std::size_t kpoint_count_;
    std::size_t current_kpoint_;
    std::size_t local_matrix_size_;
    std::vector<std::size_t> overlap_slots_;
    std::vector<TK> projected_hamiltonian_;
    std::vector<TK> projected_overlap_;
    std::vector<unsigned char> overlap_initialized_;
};

using FdeProjectedHamiltonian = FdeProjectedHamiltonianT<double>;
using FdeProjectedHamiltonianComplex
    = FdeProjectedHamiltonianT<std::complex<double>>;

} // namespace fde

#endif // FDE_EMBEDDING_PROJECTED_HAMILTONIAN_H
