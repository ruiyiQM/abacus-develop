#ifndef FDE_PROJECTED_HAMILTONIAN_H
#define FDE_PROJECTED_HAMILTONIAN_H

#include "source_hamilt/hamilt.h"

#include <cstddef>
#include <vector>

namespace fde
{

/**
 * Replicated real-Gamma Hamiltonian view whose low-energy variational space is
 * exactly the selected fragment AO principal subspace.
 */
class FdeProjectedHamiltonian : public hamilt::Hamilt<double>
{
  public:
    FdeProjectedHamiltonian(hamilt::Hamilt<double>& full_hamiltonian,
                            std::size_t full_dimension,
                            const std::vector<std::size_t>& active_orbitals,
                            double inactive_energy_ry);

    void updateHk(int kpoint) override;
    void refresh(bool yes) override;
    void matrix(hamilt::MatrixBlock<double>& hamiltonian,
                hamilt::MatrixBlock<double>& overlap) override;

    const std::vector<std::size_t>& active_orbitals() const;

  private:
    hamilt::Hamilt<double>* full_hamiltonian_;
    std::size_t full_dimension_;
    std::vector<std::size_t> active_orbitals_;
    std::vector<bool> is_active_;
    double inactive_energy_ry_;
    std::vector<double> projected_hamiltonian_;
    std::vector<double> projected_overlap_;
};

} // namespace fde

#endif // FDE_PROJECTED_HAMILTONIAN_H
