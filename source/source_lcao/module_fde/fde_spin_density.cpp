#include "fde_spin_density.h"

#include "fde_pw_pool_collectives.h"
#include "source_basis/module_pw/pw_basis.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fde
{

namespace
{

void prepare_density(double* density,
                     const std::size_t local_size,
                     const bool clamp_negative,
                     double& local_sum,
                     double& invalid)
{
    local_sum = 0.0;
    for (std::size_t point = 0; point < local_size; ++point)
    {
        if (!std::isfinite(density[point]))
        {
            invalid = 1.0;
            density[point] = 0.0;
        }
        if (clamp_negative)
        {
            density[point] = std::max(0.0, density[point]);
        }
        local_sum += density[point];
    }
}

void apply_population(double* density, const std::size_t local_size, const int electron_count, const double integral)
{
    if (electron_count == 0)
    {
        for (std::size_t point = 0; point < local_size; ++point)
        {
            density[point] = 0.0;
        }
        return;
    }
    if (!std::isfinite(integral) || integral <= std::numeric_limits<double>::min())
    {
        throw std::runtime_error("FDE spin density has zero integrated population");
    }
    const double scale = static_cast<double>(electron_count) / integral;
    for (std::size_t point = 0; point < local_size; ++point)
    {
        density[point] *= scale;
    }
}

void normalize_spin_density_impl(double* alpha_density,
                                 double* beta_density,
                                 const std::size_t local_size,
                                 const int alpha_electrons,
                                 const int beta_electrons,
                                 const double cell_volume_bohr3,
                                 const ModulePW::PW_Basis& basis,
                                 const bool clamp_negative)
{
    if (local_size != static_cast<std::size_t>(basis.nrxx) || basis.nxyz <= 0
        || (local_size != 0 && (alpha_density == nullptr || beta_density == nullptr)) || alpha_electrons < 0
        || beta_electrons < 0 || !std::isfinite(cell_volume_bohr3) || cell_volume_bohr3 <= 0.0)
    {
        throw std::invalid_argument("FDE spin-density normalization inputs are invalid");
    }

    double invalid = 0.0;
    double sums[2] = {0.0, 0.0};
    prepare_density(alpha_density, local_size, clamp_negative, sums[0], invalid);
    prepare_density(beta_density, local_size, clamp_negative, sums[1], invalid);
    invalid = PwPoolCollectives::maximum(invalid, basis);
    PwPoolCollectives::sum_in_place(sums, 2, basis);
    if (invalid != 0.0)
    {
        throw std::runtime_error("FDE spin density contains a non-finite value");
    }

    const double volume_element = cell_volume_bohr3 / static_cast<double>(basis.nxyz);
    apply_population(alpha_density, local_size, alpha_electrons, sums[0] * volume_element);
    apply_population(beta_density, local_size, beta_electrons, sums[1] * volume_element);
}

} // namespace

void normalize_spin_density(double* alpha_density,
                            double* beta_density,
                            const std::size_t local_size,
                            const int alpha_electrons,
                            const int beta_electrons,
                            const double cell_volume_bohr3,
                            const ModulePW::PW_Basis& basis)
{
    normalize_spin_density_impl(alpha_density,
                                beta_density,
                                local_size,
                                alpha_electrons,
                                beta_electrons,
                                cell_volume_bohr3,
                                basis,
                                false);
}

void normalize_nonnegative_spin_density(double* alpha_density,
                                        double* beta_density,
                                        const std::size_t local_size,
                                        const int alpha_electrons,
                                        const int beta_electrons,
                                        const double cell_volume_bohr3,
                                        const ModulePW::PW_Basis& basis)
{
    normalize_spin_density_impl(alpha_density,
                                beta_density,
                                local_size,
                                alpha_electrons,
                                beta_electrons,
                                cell_volume_bohr3,
                                basis,
                                true);
}

} // namespace fde
