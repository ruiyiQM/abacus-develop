#ifndef FDE_SPIN_DENSITY_H
#define FDE_SPIN_DENSITY_H

#include <cstddef>

namespace ModulePW
{
class PW_Basis;
}

namespace fde
{

/** Normalize local PW density slabs without changing their pointwise shape. */
void normalize_spin_density(double* alpha_density,
                            double* beta_density,
                            std::size_t local_size,
                            int alpha_electrons,
                            int beta_electrons,
                            double cell_volume_bohr3,
                            double population_tolerance,
                            const ModulePW::PW_Basis& basis);

/** Clamp negative values before normalizing final FDE density artifacts. */
void normalize_nonnegative_spin_density(double* alpha_density,
                                        double* beta_density,
                                        std::size_t local_size,
                                        int alpha_electrons,
                                        int beta_electrons,
                                        double cell_volume_bohr3,
                                        const ModulePW::PW_Basis& basis);

} // namespace fde

#endif // FDE_SPIN_DENSITY_H
