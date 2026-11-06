#ifndef FDE_TRANSITION_ENERGY_COMPONENTS_H
#define FDE_TRANSITION_ENERGY_COMPONENTS_H

#include <cstddef>
#include <vector>

namespace fde
{

/** Tr(P h), without conjugating P; column-major AO matrices, h in Ry. */
template <typename Scalar>
Scalar one_body_energy_ry(const std::vector<Scalar>& density,
                          const std::vector<Scalar>& hamiltonian_ry,
                          std::size_t dimension);

/** Local 1/2 integral rho V_H, without conjugation; total spin density, V_H in Ry.
 * The caller supplies a common Coulomb/G=0 convention and reduces across its grid pool.
 */
template <typename Scalar>
Scalar local_hartree_energy_ry(const std::vector<Scalar>& density_bohr3,
                               const std::vector<Scalar>& potential_ry,
                               double volume_element_bohr3);

} // namespace fde

#endif // FDE_TRANSITION_ENERGY_COMPONENTS_H
