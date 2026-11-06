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

} // namespace fde

#endif // FDE_TRANSITION_ENERGY_COMPONENTS_H
