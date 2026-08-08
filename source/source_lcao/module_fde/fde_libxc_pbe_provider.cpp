#include "fde_potential_evaluator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef __LIBXC
#include <xc.h>
#include <xc_funcs.h>
#endif

namespace fde
{

#ifdef __LIBXC
namespace
{

struct GridFunctionalResult
{
    double energy_hartree;
    std::vector<double> alpha_potential_hartree;
    std::vector<double> beta_potential_hartree;
};

std::size_t checked_grid_size(const UniformGrid& grid)
{
    if (grid.x == 0 || grid.y == 0 || grid.z == 0
        || !std::isfinite(grid.spacing_x_bohr) || grid.spacing_x_bohr <= 0.0
        || !std::isfinite(grid.spacing_y_bohr) || grid.spacing_y_bohr <= 0.0
        || !std::isfinite(grid.spacing_z_bohr) || grid.spacing_z_bohr <= 0.0)
    {
        throw std::invalid_argument("Libxc PBE requires a positive orthorhombic uniform grid");
    }
    return grid.x * grid.y * grid.z;
}

std::size_t offset(const std::size_t x,
                   const std::size_t y,
                   const std::size_t z,
                   const UniformGrid& grid)
{
    return x + grid.x * (y + grid.y * z);
}

std::size_t previous(const std::size_t index, const std::size_t extent)
{
    return index == 0 ? extent - 1 : index - 1;
}

std::size_t next(const std::size_t index, const std::size_t extent)
{
    return index + 1 == extent ? 0 : index + 1;
}

void gradient(const std::vector<double>& values,
              const UniformGrid& grid,
              std::vector<double>& dx,
              std::vector<double>& dy,
              std::vector<double>& dz)
{
    dx.assign(values.size(), 0.0);
    dy.assign(values.size(), 0.0);
    dz.assign(values.size(), 0.0);
    for (std::size_t iz = 0; iz < grid.z; ++iz)
    {
        for (std::size_t iy = 0; iy < grid.y; ++iy)
        {
            for (std::size_t ix = 0; ix < grid.x; ++ix)
            {
                const std::size_t center = offset(ix, iy, iz, grid);
                dx[center] = (values[offset(next(ix, grid.x), iy, iz, grid)]
                              - values[offset(previous(ix, grid.x), iy, iz, grid)])
                             / (2.0 * grid.spacing_x_bohr);
                dy[center] = (values[offset(ix, next(iy, grid.y), iz, grid)]
                              - values[offset(ix, previous(iy, grid.y), iz, grid)])
                             / (2.0 * grid.spacing_y_bohr);
                dz[center] = (values[offset(ix, iy, next(iz, grid.z), grid)]
                              - values[offset(ix, iy, previous(iz, grid.z), grid)])
                             / (2.0 * grid.spacing_z_bohr);
            }
        }
    }
}

std::vector<double> divergence(const std::vector<double>& vx,
                               const std::vector<double>& vy,
                               const std::vector<double>& vz,
                               const UniformGrid& grid)
{
    std::vector<double> result(vx.size(), 0.0);
    for (std::size_t iz = 0; iz < grid.z; ++iz)
    {
        for (std::size_t iy = 0; iy < grid.y; ++iy)
        {
            for (std::size_t ix = 0; ix < grid.x; ++ix)
            {
                const std::size_t center = offset(ix, iy, iz, grid);
                result[center]
                    = (vx[offset(next(ix, grid.x), iy, iz, grid)]
                       - vx[offset(previous(ix, grid.x), iy, iz, grid)])
                          / (2.0 * grid.spacing_x_bohr)
                      + (vy[offset(ix, next(iy, grid.y), iz, grid)]
                         - vy[offset(ix, previous(iy, grid.y), iz, grid)])
                            / (2.0 * grid.spacing_y_bohr)
                      + (vz[offset(ix, iy, next(iz, grid.z), grid)]
                         - vz[offset(ix, iy, previous(iz, grid.z), grid)])
                            / (2.0 * grid.spacing_z_bohr);
            }
        }
    }
    return result;
}

void validate_density(const SpinDensity& density, const std::size_t size)
{
    if (density.alpha_bohr3.size() != size || density.beta_bohr3.size() != size)
    {
        throw std::invalid_argument("Libxc PBE density does not match the uniform grid");
    }
    for (std::size_t index = 0; index < size; ++index)
    {
        if (!std::isfinite(density.alpha_bohr3[index]) || density.alpha_bohr3[index] < 0.0
            || !std::isfinite(density.beta_bohr3[index]) || density.beta_bohr3[index] < 0.0)
        {
            throw std::invalid_argument("Libxc PBE densities must be finite and nonnegative");
        }
    }
}

GridFunctionalResult evaluate_pbe(const SpinDensity& density,
                                  const UniformGrid& grid,
                                  const double density_floor)
{
    const std::size_t size = checked_grid_size(grid);
    validate_density(density, size);

    std::vector<double> alpha(size, density_floor);
    std::vector<double> beta(size, density_floor);
    for (std::size_t index = 0; index < size; ++index)
    {
        alpha[index] = std::max(density.alpha_bohr3[index], density_floor);
        beta[index] = std::max(density.beta_bohr3[index], density_floor);
    }

    std::vector<double> dax;
    std::vector<double> day;
    std::vector<double> daz;
    std::vector<double> dbx;
    std::vector<double> dby;
    std::vector<double> dbz;
    gradient(alpha, grid, dax, day, daz);
    gradient(beta, grid, dbx, dby, dbz);

    std::vector<double> rho(2 * size, 0.0);
    std::vector<double> sigma(3 * size, 0.0);
    for (std::size_t index = 0; index < size; ++index)
    {
        rho[2 * index] = alpha[index];
        rho[2 * index + 1] = beta[index];
        sigma[3 * index] = dax[index] * dax[index] + day[index] * day[index]
                           + daz[index] * daz[index];
        sigma[3 * index + 1] = dax[index] * dbx[index] + day[index] * dby[index]
                               + daz[index] * dbz[index];
        sigma[3 * index + 2] = dbx[index] * dbx[index] + dby[index] * dby[index]
                               + dbz[index] * dbz[index];
    }

    const int functional_ids[2] = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    std::vector<double> energy_per_particle(size, 0.0);
    std::vector<double> vrho(2 * size, 0.0);
    std::vector<double> vsigma(3 * size, 0.0);
    double reference_energy_density = 0.0;
    for (int functional_index = 0; functional_index < 2; ++functional_index)
    {
        xc_func_type functional = {};
        if (xc_func_init(&functional, functional_ids[functional_index], XC_POLARIZED) != 0)
        {
            throw std::runtime_error("Libxc failed to initialize a PBE functional");
        }
        std::vector<double> local_energy(size, 0.0);
        std::vector<double> local_vrho(2 * size, 0.0);
        std::vector<double> local_vsigma(3 * size, 0.0);
        xc_gga_exc_vxc(&functional,
                       size,
                       rho.data(),
                       sigma.data(),
                       local_energy.data(),
                       local_vrho.data(),
                       local_vsigma.data());

        const double reference_rho[2] = {density_floor, density_floor};
        const double reference_sigma[3] = {0.0, 0.0, 0.0};
        double reference_energy = 0.0;
        double reference_vrho[2] = {0.0, 0.0};
        double reference_vsigma[3] = {0.0, 0.0, 0.0};
        xc_gga_exc_vxc(&functional,
                       1,
                       reference_rho,
                       reference_sigma,
                       &reference_energy,
                       reference_vrho,
                       reference_vsigma);
        reference_energy_density += 2.0 * density_floor * reference_energy;
        xc_func_end(&functional);

        for (std::size_t index = 0; index < size; ++index)
        {
            energy_per_particle[index] += local_energy[index];
        }
        for (std::size_t index = 0; index < 2 * size; ++index)
        {
            vrho[index] += local_vrho[index];
        }
        for (std::size_t index = 0; index < 3 * size; ++index)
        {
            vsigma[index] += local_vsigma[index];
        }
    }

    std::vector<double> fax(size, 0.0);
    std::vector<double> fay(size, 0.0);
    std::vector<double> faz(size, 0.0);
    std::vector<double> fbx(size, 0.0);
    std::vector<double> fby(size, 0.0);
    std::vector<double> fbz(size, 0.0);
    GridFunctionalResult result;
    result.energy_hartree = 0.0;
    result.alpha_potential_hartree.resize(size);
    result.beta_potential_hartree.resize(size);
    const double volume = grid.spacing_x_bohr * grid.spacing_y_bohr * grid.spacing_z_bohr;
    for (std::size_t index = 0; index < size; ++index)
    {
        result.energy_hartree
            += ((alpha[index] + beta[index]) * energy_per_particle[index]
                - reference_energy_density)
               * volume;
        const double aa = 2.0 * vsigma[3 * index];
        const double ab = vsigma[3 * index + 1];
        const double bb = 2.0 * vsigma[3 * index + 2];
        fax[index] = aa * dax[index] + ab * dbx[index];
        fay[index] = aa * day[index] + ab * dby[index];
        faz[index] = aa * daz[index] + ab * dbz[index];
        fbx[index] = ab * dax[index] + bb * dbx[index];
        fby[index] = ab * day[index] + bb * dby[index];
        fbz[index] = ab * daz[index] + bb * dbz[index];
    }
    const std::vector<double> divergence_alpha = divergence(fax, fay, faz, grid);
    const std::vector<double> divergence_beta = divergence(fbx, fby, fbz, grid);
    for (std::size_t index = 0; index < size; ++index)
    {
        result.alpha_potential_hartree[index]
            = density.alpha_bohr3[index] > density_floor
                  ? vrho[2 * index] - divergence_alpha[index]
                  : 0.0;
        result.beta_potential_hartree[index]
            = density.beta_bohr3[index] > density_floor
                  ? vrho[2 * index + 1] - divergence_beta[index]
                  : 0.0;
    }
    return result;
}

SpinDensity sum_density(const SpinDensity& first, const SpinDensity& second)
{
    SpinDensity result;
    result.alpha_bohr3.resize(first.alpha_bohr3.size());
    result.beta_bohr3.resize(first.beta_bohr3.size());
    for (std::size_t index = 0; index < first.alpha_bohr3.size(); ++index)
    {
        result.alpha_bohr3[index] = first.alpha_bohr3[index] + second.alpha_bohr3[index];
        result.beta_bohr3[index] = first.beta_bohr3[index] + second.beta_bohr3[index];
    }
    return result;
}

} // namespace
#endif

bool LibxcPbeProvider::available()
{
#ifdef __LIBXC
    return true;
#else
    return false;
#endif
}

NonadditiveFunctionalResult LibxcPbeProvider::evaluate(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3) const
{
#ifdef __LIBXC
    if (!std::isfinite(density_floor_bohr3) || density_floor_bohr3 <= 0.0)
    {
        throw std::invalid_argument("Libxc PBE density floor must be finite and positive");
    }
    const std::size_t size = checked_grid_size(grid);
    validate_density(active, size);
    validate_density(frozen, size);
    const GridFunctionalResult total = evaluate_pbe(sum_density(active, frozen),
                                                    grid,
                                                    density_floor_bohr3);
    const GridFunctionalResult active_only = evaluate_pbe(active, grid, density_floor_bohr3);
    const GridFunctionalResult frozen_only = evaluate_pbe(frozen, grid, density_floor_bohr3);

    const double hartree_to_rydberg = 2.0;
    NonadditiveFunctionalResult result;
    result.energy_ry = hartree_to_rydberg
                       * (total.energy_hartree - active_only.energy_hartree
                          - frozen_only.energy_hartree);
    result.active_potential.alpha_ry.resize(size);
    result.active_potential.beta_ry.resize(size);
    result.frozen_potential.alpha_ry.resize(size);
    result.frozen_potential.beta_ry.resize(size);
    for (std::size_t index = 0; index < size; ++index)
    {
        result.active_potential.alpha_ry[index]
            = hartree_to_rydberg
              * (total.alpha_potential_hartree[index]
                 - active_only.alpha_potential_hartree[index]);
        result.active_potential.beta_ry[index]
            = hartree_to_rydberg
              * (total.beta_potential_hartree[index]
                 - active_only.beta_potential_hartree[index]);
        result.frozen_potential.alpha_ry[index]
            = hartree_to_rydberg
              * (total.alpha_potential_hartree[index]
                 - frozen_only.alpha_potential_hartree[index]);
        result.frozen_potential.beta_ry[index]
            = hartree_to_rydberg
              * (total.beta_potential_hartree[index]
                 - frozen_only.beta_potential_hartree[index]);
    }
    return result;
#else
    (void)active;
    (void)frozen;
    (void)grid;
    (void)density_floor_bohr3;
    throw std::runtime_error("Libxc PBE support is unavailable in this ABACUS build");
#endif
}

} // namespace fde
