#include "fde_potential_evaluator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

const std::size_t libxc_chunk_size = 32768;

int maximum_thread_count()
{
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

int current_thread_index()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

class LibxcFunctionalPool
{
  public:
    LibxcFunctionalPool(const int functional_id, const int count)
        : functionals_(static_cast<std::size_t>(count)), initialized_(0)
    {
        if (count <= 0)
        {
            throw std::invalid_argument("Libxc functional pool must be nonempty");
        }
        for (int index = 0; index < count; ++index)
        {
            if (xc_func_init(&functionals_[index], functional_id, XC_POLARIZED) != 0)
            {
                for (int initialized = 0; initialized < initialized_; ++initialized)
                {
                    xc_func_end(&functionals_[initialized]);
                }
                throw std::runtime_error("Libxc failed to initialize a PBE functional");
            }
            ++initialized_;
        }
    }

    ~LibxcFunctionalPool()
    {
        for (int index = 0; index < initialized_; ++index)
        {
            xc_func_end(&functionals_[index]);
        }
    }

    xc_func_type& operator[](const int index)
    {
        return functionals_[static_cast<std::size_t>(index)];
    }

  private:
    std::vector<xc_func_type> functionals_;
    int initialized_;

    LibxcFunctionalPool(const LibxcFunctionalPool&);
    LibxcFunctionalPool& operator=(const LibxcFunctionalPool&);
};

struct LibxcThreadScratch
{
    explicit LibxcThreadScratch(const std::size_t capacity)
        : energy(capacity, 0.0), vrho(2 * capacity, 0.0), vsigma(3 * capacity, 0.0)
    {
    }

    std::vector<double> energy;
    std::vector<double> vrho;
    std::vector<double> vsigma;
};

double accumulate_functional(const int functional_id,
                             const std::vector<double>& rho,
                             const std::vector<double>& sigma,
                             const double density_floor,
                             std::vector<double>& energy_per_particle,
                             std::vector<double>& vrho,
                             std::vector<double>& vsigma)
{
    const std::size_t size = energy_per_particle.size();
    const int thread_count = maximum_thread_count();
    LibxcFunctionalPool functionals(functional_id, thread_count);
    std::vector<LibxcThreadScratch> scratch;
    scratch.reserve(static_cast<std::size_t>(thread_count));
    for (int thread = 0; thread < thread_count; ++thread)
    {
        scratch.push_back(LibxcThreadScratch(libxc_chunk_size));
    }

    const double reference_rho[2] = {density_floor, density_floor};
    const double reference_sigma[3] = {0.0, 0.0, 0.0};
    double reference_energy = 0.0;
    double reference_vrho[2] = {0.0, 0.0};
    double reference_vsigma[3] = {0.0, 0.0, 0.0};
    xc_gga_exc_vxc(&functionals[0],
                   1,
                   reference_rho,
                   reference_sigma,
                   &reference_energy,
                   reference_vrho,
                   reference_vsigma);

    const std::size_t chunk_count
        = (size + libxc_chunk_size - 1) / libxc_chunk_size;
#pragma omp parallel for schedule(static)
    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk)
    {
        const int thread = current_thread_index();
        LibxcThreadScratch& local = scratch[static_cast<std::size_t>(thread)];
        const std::size_t begin = chunk * libxc_chunk_size;
        const std::size_t count = std::min(libxc_chunk_size, size - begin);
        xc_gga_exc_vxc(&functionals[thread],
                       count,
                       &rho[2 * begin],
                       &sigma[3 * begin],
                       local.energy.data(),
                       local.vrho.data(),
                       local.vsigma.data());
        for (std::size_t local_index = 0; local_index < count; ++local_index)
        {
            const std::size_t index = begin + local_index;
            energy_per_particle[index] += local.energy[local_index];
            vrho[2 * index] += local.vrho[2 * local_index];
            vrho[2 * index + 1] += local.vrho[2 * local_index + 1];
            vsigma[3 * index] += local.vsigma[3 * local_index];
            vsigma[3 * index + 1] += local.vsigma[3 * local_index + 1];
            vsigma[3 * index + 2] += local.vsigma[3 * local_index + 2];
        }
    }
    return 2.0 * density_floor * reference_energy;
}

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
    return (x * grid.y + y) * grid.z + z;
}

std::size_t previous(const std::size_t index, const std::size_t extent)
{
    return index == 0 ? extent - 1 : index - 1;
}

std::size_t next(const std::size_t index, const std::size_t extent)
{
    return index + 1 == extent ? 0 : index + 1;
}

void finite_difference_gradient(const std::vector<double>& values,
                                const UniformGrid& grid,
                                std::vector<double>& dx,
                                std::vector<double>& dy,
                                std::vector<double>& dz)
{
    dx.assign(values.size(), 0.0);
    dy.assign(values.size(), 0.0);
    dz.assign(values.size(), 0.0);
#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t ix = 0; ix < grid.x; ++ix)
    {
        for (std::size_t iy = 0; iy < grid.y; ++iy)
        {
            for (std::size_t iz = 0; iz < grid.z; ++iz)
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

std::vector<double> finite_difference_divergence(const std::vector<double>& vx,
                                                 const std::vector<double>& vy,
                                                 const std::vector<double>& vz,
                                                 const UniformGrid& grid)
{
    std::vector<double> result(vx.size(), 0.0);
#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t ix = 0; ix < grid.x; ++ix)
    {
        for (std::size_t iy = 0; iy < grid.y; ++iy)
        {
            for (std::size_t iz = 0; iz < grid.z; ++iz)
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

void apply_gradient(const std::vector<double>& values,
                    const UniformGrid& grid,
                    const GridDifferentialOperator* differential_operator,
                    std::vector<double>& dx,
                    std::vector<double>& dy,
                    std::vector<double>& dz)
{
    if (differential_operator == nullptr)
    {
        finite_difference_gradient(values, grid, dx, dy, dz);
        return;
    }
    differential_operator->gradient(values, dx, dy, dz);
}

std::vector<double> apply_divergence(
    const std::vector<double>& vx,
    const std::vector<double>& vy,
    const std::vector<double>& vz,
    const UniformGrid& grid,
    const GridDifferentialOperator* differential_operator)
{
    return differential_operator == nullptr
               ? finite_difference_divergence(vx, vy, vz, grid)
               : differential_operator->divergence(vx, vy, vz);
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
                                  const double density_floor,
                                  const GridDifferentialOperator* differential_operator)
{
    const std::size_t global_size = checked_grid_size(grid);
    const std::size_t size = differential_operator == nullptr
                                 ? global_size
                                 : differential_operator->local_size();
    validate_density(density, size);

    std::vector<double> alpha(size, density_floor);
    std::vector<double> beta(size, density_floor);
#pragma omp parallel for schedule(static)
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
    apply_gradient(alpha, grid, differential_operator, dax, day, daz);
    apply_gradient(beta, grid, differential_operator, dbx, dby, dbz);

    std::vector<double> rho(2 * size, 0.0);
    std::vector<double> sigma(3 * size, 0.0);
#pragma omp parallel for schedule(static)
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
        reference_energy_density
            += accumulate_functional(functional_ids[functional_index],
                                     rho,
                                     sigma,
                                     density_floor,
                                     energy_per_particle,
                                     vrho,
                                     vsigma);
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
    double energy_hartree = 0.0;
#pragma omp parallel for reduction(+:energy_hartree) schedule(static)
    for (std::size_t index = 0; index < size; ++index)
    {
        energy_hartree
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
    result.energy_hartree = energy_hartree;
    const std::vector<double> divergence_alpha
        = apply_divergence(fax, fay, faz, grid, differential_operator);
    const std::vector<double> divergence_beta
        = apply_divergence(fbx, fby, fbz, grid, differential_operator);
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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

FrozenSemilocalCache LibxcPbeProvider::prepare_frozen(
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3,
    const GridDifferentialOperator* differential_operator) const
{
#ifdef __LIBXC
    if (!std::isfinite(density_floor_bohr3) || density_floor_bohr3 <= 0.0)
    {
        throw std::invalid_argument("Libxc PBE density floor must be finite and positive");
    }
    const std::size_t global_size = checked_grid_size(grid);
    const std::size_t size = differential_operator == nullptr
                                 ? global_size
                                 : differential_operator->local_size();
    validate_density(frozen, size);
    const GridFunctionalResult frozen_only
        = evaluate_pbe(frozen, grid, density_floor_bohr3, differential_operator);
    const double hartree_to_rydberg = 2.0;
    FrozenSemilocalCache cache;
    cache.energy_ry = hartree_to_rydberg * frozen_only.energy_hartree;
    cache.potential_ry.alpha_ry.resize(size);
    cache.potential_ry.beta_ry.resize(size);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < size; ++index)
    {
        cache.potential_ry.alpha_ry[index]
            = hartree_to_rydberg * frozen_only.alpha_potential_hartree[index];
        cache.potential_ry.beta_ry[index]
            = hartree_to_rydberg * frozen_only.beta_potential_hartree[index];
    }
    return cache;
#else
    (void)frozen;
    (void)grid;
    (void)density_floor_bohr3;
    (void)differential_operator;
    throw std::runtime_error("Libxc PBE support is unavailable in this ABACUS build");
#endif
}

NonadditiveFunctionalResult LibxcPbeProvider::evaluate(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3,
    const GridDifferentialOperator* differential_operator) const
{
#ifdef __LIBXC
    const FrozenSemilocalCache frozen_cache
        = prepare_frozen(frozen, grid, density_floor_bohr3, differential_operator);
    return evaluate_cached(active,
                           frozen,
                           grid,
                           density_floor_bohr3,
                           frozen_cache,
                           differential_operator);
#else
    (void)active;
    (void)frozen;
    (void)grid;
    (void)density_floor_bohr3;
    (void)differential_operator;
    throw std::runtime_error("Libxc PBE support is unavailable in this ABACUS build");
#endif
}

NonadditiveFunctionalResult LibxcPbeProvider::evaluate_cached(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3,
    const FrozenSemilocalCache& frozen_cache,
    const GridDifferentialOperator* differential_operator) const
{
#ifdef __LIBXC
    if (!std::isfinite(density_floor_bohr3) || density_floor_bohr3 <= 0.0)
    {
        throw std::invalid_argument("Libxc PBE density floor must be finite and positive");
    }
    const std::size_t global_size = checked_grid_size(grid);
    const std::size_t size = differential_operator == nullptr
                                 ? global_size
                                 : differential_operator->local_size();
    validate_density(active, size);
    validate_density(frozen, size);
    if (!std::isfinite(frozen_cache.energy_ry)
        || frozen_cache.potential_ry.alpha_ry.size() != size
        || frozen_cache.potential_ry.beta_ry.size() != size)
    {
        throw std::invalid_argument("Libxc PBE frozen cache does not match the grid");
    }
    const GridFunctionalResult total = evaluate_pbe(sum_density(active, frozen),
                                                    grid,
                                                    density_floor_bohr3,
                                                    differential_operator);
    const GridFunctionalResult active_only
        = evaluate_pbe(active, grid, density_floor_bohr3, differential_operator);

    const double hartree_to_rydberg = 2.0;
    NonadditiveFunctionalResult result;
    result.energy_ry = hartree_to_rydberg
                       * (total.energy_hartree - active_only.energy_hartree)
                       - frozen_cache.energy_ry;
    result.active_potential.alpha_ry.resize(size);
    result.active_potential.beta_ry.resize(size);
    result.frozen_potential.alpha_ry.resize(size);
    result.frozen_potential.beta_ry.resize(size);
#pragma omp parallel for schedule(static)
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
            = hartree_to_rydberg * total.alpha_potential_hartree[index]
              - frozen_cache.potential_ry.alpha_ry[index];
        result.frozen_potential.beta_ry[index]
            = hartree_to_rydberg * total.beta_potential_hartree[index]
              - frozen_cache.potential_ry.beta_ry[index];
    }
    return result;
#else
    (void)active;
    (void)frozen;
    (void)grid;
    (void)density_floor_bohr3;
    (void)frozen_cache;
    (void)differential_operator;
    throw std::runtime_error("Libxc PBE support is unavailable in this ABACUS build");
#endif
}

} // namespace fde
