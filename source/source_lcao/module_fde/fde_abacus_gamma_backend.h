#ifndef FDE_ABACUS_GAMMA_BACKEND_H
#define FDE_ABACUS_GAMMA_BACKEND_H

#include "fde_one_way_scf.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace hamilt
{
template <typename T>
class HContainer;
}

namespace fde
{

/**
 * Narrow integration seam around ABACUS grid integration. Keeping this seam
 * explicit makes the FDE SCF independent of Gint's process-wide setup and lets
 * the AO/grid conversion be tested without initializing a complete calculation.
 */
class GammaGridIntegrator
{
  public:
    virtual ~GammaGridIntegrator() {}

    virtual void potential_to_ao(const double* potential_ry,
                                 hamilt::HContainer<double>& matrix) const = 0;

    virtual void density_to_grid(const std::vector<hamilt::HContainer<double>*>& density_matrices,
                                 const int spin_channels,
                                 double** density_bohr3) const = 0;
};

/** Real ABACUS Gint implementation of GammaGridIntegrator. */
class AbacusGintGammaIntegrator : public GammaGridIntegrator
{
  public:
    void potential_to_ao(const double* potential_ry,
                         hamilt::HContainer<double>& matrix) const override;

    void density_to_grid(const std::vector<hamilt::HContainer<double>*>& density_matrices,
                         const int spin_channels,
                         double** density_bohr3) const override;
};

/**
 * Γ-point AO/grid backend for the current dense-matrix OneWayScf contract.
 *
 * The current contract stores a complete dense AO matrix, so this adapter
 * deliberately rejects a distributed 2D-block AO container. A later distributed
 * driver can replace the dense contract without changing the FDE equations.
 */
class AbacusGammaBackend : public OneWayScfBackend
{
  public:
    AbacusGammaBackend(const hamilt::HContainer<double>& gamma_structure,
                       const std::size_t local_grid_size,
                       const std::shared_ptr<const GammaGridIntegrator>& integrator);

    SpinAoMatrix embedding_potential_matrix(
        const SpinPotential& real_space_potential,
        const std::size_t full_ao_dimension) const override;

    SpinDensity density_from_ao_matrices(
        const SpinAoMatrix& density_matrices,
        const std::size_t full_ao_dimension,
        const UniformGrid& grid) const override;

  private:
    void validate_ao_contract(const std::size_t full_ao_dimension) const;

    const hamilt::HContainer<double>* gamma_structure_;
    std::size_t local_grid_size_;
    std::shared_ptr<const GammaGridIntegrator> integrator_;
};

} // namespace fde

#endif // FDE_ABACUS_GAMMA_BACKEND_H
