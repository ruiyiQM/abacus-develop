#include "fde_semilocal_functional.h"

#ifdef __CUDA
#include "fde_gpu_kernels.h"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fde
{

namespace
{

const double pi = 3.141592653589793238462643383279502884;
const double hartree_to_rydberg = 2.0;

struct ScalarFunctionalResult
{
    double energy_hartree;
    std::vector<double> potential_hartree;
};

std::size_t grid_size(const UniformGrid& grid)
{
    if (grid.x == 0 || grid.y == 0 || grid.z == 0
        || !std::isfinite(grid.spacing_x_bohr) || grid.spacing_x_bohr <= 0.0
        || !std::isfinite(grid.spacing_y_bohr) || grid.spacing_y_bohr <= 0.0
        || !std::isfinite(grid.spacing_z_bohr) || grid.spacing_z_bohr <= 0.0)
    {
        throw std::invalid_argument("FDE uniform grid dimensions and spacings must be positive");
    }
    return grid.x * grid.y * grid.z;
}

std::size_t evaluation_size(const UniformGrid& grid,
                            const GridDifferentialOperator* differential_operator)
{
    const std::size_t global_size = grid_size(grid);
    return differential_operator == nullptr ? global_size
                                            : differential_operator->local_size();
}

std::size_t offset(const std::size_t x,
                   const std::size_t y,
                   const std::size_t z,
                   const UniformGrid& grid)
{
    return (x * grid.y + y) * grid.z + z;
}

std::size_t periodic_previous(const std::size_t index, const std::size_t extent)
{
    return index == 0 ? extent - 1 : index - 1;
}

std::size_t periodic_next(const std::size_t index, const std::size_t extent)
{
    return index + 1 == extent ? 0 : index + 1;
}

void validate_density(const SpinDensity& density, const std::size_t expected_size)
{
    if (density.alpha_bohr3.size() != expected_size || density.beta_bohr3.size() != expected_size)
    {
        throw std::invalid_argument("FDE spin-density arrays do not match the uniform grid");
    }
    for (std::size_t index = 0; index < expected_size; ++index)
    {
        if (!std::isfinite(density.alpha_bohr3[index]) || density.alpha_bohr3[index] < 0.0
            || !std::isfinite(density.beta_bohr3[index]) || density.beta_bohr3[index] < 0.0)
        {
            throw std::invalid_argument("FDE spin densities must be finite and nonnegative");
        }
    }
}

bool equal_spin_channels(
    const SpinDensity& density,
    const GridDifferentialOperator* differential_operator)
{
    const bool locally_equal
        = density.alpha_bohr3 == density.beta_bohr3;
    return differential_operator == nullptr
               ? locally_equal
               : differential_operator->all_processes(locally_equal);
}

void validate_frozen_cache(const FrozenSemilocalCache& cache,
                           const std::size_t expected_size)
{
    if (!std::isfinite(cache.energy_ry)
        || cache.potential_ry.alpha_ry.size() != expected_size
        || cache.potential_ry.beta_ry.size() != expected_size)
    {
        throw std::invalid_argument("FDE frozen functional cache does not match the grid");
    }
}

std::vector<double> add_density(const std::vector<double>& first,
                                const std::vector<double>& second)
{
    std::vector<double> sum(first.size(), 0.0);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        sum[index] = first[index] + second[index];
    }
    return sum;
}

void finite_difference_gradient(const std::vector<double>& values,
                                const UniformGrid& grid,
                                std::vector<double>& gradient_x,
                                std::vector<double>& gradient_y,
                                std::vector<double>& gradient_z)
{
    const std::size_t size = values.size();
    gradient_x.assign(size, 0.0);
    gradient_y.assign(size, 0.0);
    gradient_z.assign(size, 0.0);
#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t x = 0; x < grid.x; ++x)
    {
        for (std::size_t y = 0; y < grid.y; ++y)
        {
            for (std::size_t z = 0; z < grid.z; ++z)
            {
                const std::size_t center = offset(x, y, z, grid);
                gradient_x[center]
                    = (values[offset(periodic_next(x, grid.x), y, z, grid)]
                       - values[offset(periodic_previous(x, grid.x), y, z, grid)])
                      / (2.0 * grid.spacing_x_bohr);
                gradient_y[center]
                    = (values[offset(x, periodic_next(y, grid.y), z, grid)]
                       - values[offset(x, periodic_previous(y, grid.y), z, grid)])
                      / (2.0 * grid.spacing_y_bohr);
                gradient_z[center]
                    = (values[offset(x, y, periodic_next(z, grid.z), grid)]
                       - values[offset(x, y, periodic_previous(z, grid.z), grid)])
                      / (2.0 * grid.spacing_z_bohr);
            }
        }
    }
}

std::vector<double> finite_difference_divergence(
    const std::vector<double>& vector_x,
    const std::vector<double>& vector_y,
    const std::vector<double>& vector_z,
    const UniformGrid& grid)
{
    std::vector<double> result(vector_x.size(), 0.0);
#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t x = 0; x < grid.x; ++x)
    {
        for (std::size_t y = 0; y < grid.y; ++y)
        {
            for (std::size_t z = 0; z < grid.z; ++z)
            {
                const std::size_t center = offset(x, y, z, grid);
                result[center]
                    = (vector_x[offset(periodic_next(x, grid.x), y, z, grid)]
                       - vector_x[offset(periodic_previous(x, grid.x), y, z, grid)])
                          / (2.0 * grid.spacing_x_bohr)
                      + (vector_y[offset(x, periodic_next(y, grid.y), z, grid)]
                         - vector_y[offset(x, periodic_previous(y, grid.y), z, grid)])
                            / (2.0 * grid.spacing_y_bohr)
                      + (vector_z[offset(x, y, periodic_next(z, grid.z), grid)]
                         - vector_z[offset(x, y, periodic_previous(z, grid.z), grid)])
                            / (2.0 * grid.spacing_z_bohr);
            }
        }
    }
    return result;
}

void apply_gradient(const std::vector<double>& values,
                    const UniformGrid& grid,
                    const GridDifferentialOperator* differential_operator,
                    std::vector<double>& gradient_x,
                    std::vector<double>& gradient_y,
                    std::vector<double>& gradient_z)
{
    if (differential_operator == nullptr)
    {
        finite_difference_gradient(values,
                                   grid,
                                   gradient_x,
                                   gradient_y,
                                   gradient_z);
        return;
    }
    differential_operator->gradient(values, gradient_x, gradient_y, gradient_z);
}

std::vector<double> apply_divergence(
    const std::vector<double>& vector_x,
    const std::vector<double>& vector_y,
    const std::vector<double>& vector_z,
    const UniformGrid& grid,
    const GridDifferentialOperator* differential_operator)
{
    return differential_operator == nullptr
               ? finite_difference_divergence(vector_x, vector_y, vector_z, grid)
               : differential_operator->divergence(vector_x, vector_y, vector_z);
}

void kinetic_enhancement(const KineticFunctional functional,
                         const double reduced_gradient,
                         double& enhancement,
                         double& derivative)
{
    if (functional == KineticFunctional::ThomasFermi)
    {
        enhancement = 1.0;
        derivative = 0.0;
        return;
    }

    if (functional == KineticFunctional::RevApbek)
    {
        const double kappa = 1.245;
        const double mu = 0.23889;
        const double s2 = reduced_gradient * reduced_gradient;
        const double denominator = kappa + mu * s2;
        enhancement = 1.0 + kappa * mu * s2 / denominator;
        derivative = 2.0 * kappa * kappa * mu * reduced_gradient
                     / (denominator * denominator);
        return;
    }

    if (functional != KineticFunctional::Pw91k)
    {
        throw std::invalid_argument("FDE received an unknown kinetic functional");
    }

    const double a = 0.093907;
    const double b = 76.320;
    const double c = 0.26608;
    const double d = -0.0809615;
    const double f = 0.000057767;
    const double alpha = 100.0;
    const double exponent = 4.0;
    const double s = reduced_gradient;
    const double s2 = s * s;
    const double exponential = std::exp(-alpha * s2);
    const double s_exponent = std::pow(s, exponent);
    const double numerator = (c + d * exponential) * s2 - f * s_exponent;
    const double denominator = 1.0 + a * s * std::asinh(b * s) + f * s_exponent;
    const double numerator_derivative
        = 2.0 * s * (c + d * exponential)
          - 2.0 * alpha * d * s * s2 * exponential
          - f * exponent * std::pow(s, exponent - 1.0);
    const double denominator_derivative
        = a * (std::asinh(b * s) + b * s / std::sqrt(1.0 + b * b * s2))
          + f * exponent * std::pow(s, exponent - 1.0);
    enhancement = 1.0 + numerator / denominator;
    derivative = (numerator_derivative * denominator
                  - numerator * denominator_derivative)
                 / (denominator * denominator);
}

ScalarFunctionalResult evaluate_unpolarized_kinetic(const std::vector<double>& density,
                                                     const UniformGrid& grid,
                                                     const KineticFunctional functional,
                                                     const double density_floor,
                                                     const GridDifferentialOperator*
                                                         differential_operator)
{
    const std::size_t size = evaluation_size(grid, differential_operator);
    std::vector<double> regularized(size, density_floor);
    std::vector<unsigned char> active(size, 0);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < size; ++index)
    {
        regularized[index] = std::max(density[index], density_floor);
        active[index] = density[index] > density_floor ? 1 : 0;
    }

    const double c_tf = 0.3 * std::pow(3.0 * pi * pi, 2.0 / 3.0);
    const double reduced_gradient_scale = 1.0 / (2.0 * std::pow(3.0 * pi * pi, 1.0 / 3.0));
    const double reference_energy_density = c_tf * std::pow(density_floor, 5.0 / 3.0);
    const double volume_element
        = grid.spacing_x_bohr * grid.spacing_y_bohr * grid.spacing_z_bohr;
    ScalarFunctionalResult result;
    result.energy_hartree = 0.0;
    result.potential_hartree.assign(size, 0.0);
    if (functional == KineticFunctional::ThomasFermi)
    {
#ifdef __CUDA
        if (differential_operator != nullptr
            && differential_operator->uses_gpu())
        {
            const std::vector<double> no_gradient;
            std::vector<double> no_flux_x;
            std::vector<double> no_flux_y;
            std::vector<double> no_flux_z;
            gpu_kinetic_local_terms(regularized,
                                    active,
                                    no_gradient,
                                    no_gradient,
                                    no_gradient,
                                    functional,
                                    density_floor,
                                    volume_element,
                                    result.potential_hartree,
                                    no_flux_x,
                                    no_flux_y,
                                    no_flux_z,
                                    result.energy_hartree);
            return result;
        }
#endif
        double energy_hartree = 0.0;
#pragma omp parallel for reduction(+:energy_hartree) schedule(static)
        for (std::size_t index = 0; index < size; ++index)
        {
            const double rho = regularized[index];
            energy_hartree
                += (c_tf * std::pow(rho, 5.0 / 3.0) - reference_energy_density)
                   * volume_element;
            if (active[index])
            {
                result.potential_hartree[index]
                    = (5.0 / 3.0) * c_tf * std::pow(rho, 2.0 / 3.0);
            }
        }
        result.energy_hartree = energy_hartree;
        return result;
    }

    std::vector<double> gradient_x;
    std::vector<double> gradient_y;
    std::vector<double> gradient_z;
    apply_gradient(regularized,
                   grid,
                   differential_operator,
                   gradient_x,
                   gradient_y,
                   gradient_z);
    std::vector<double> flux_x(size, 0.0);
    std::vector<double> flux_y(size, 0.0);
    std::vector<double> flux_z(size, 0.0);

    double energy_hartree = 0.0;
#ifdef __CUDA
    const bool evaluate_on_gpu = differential_operator != nullptr
                                 && differential_operator->uses_gpu();
    if (evaluate_on_gpu)
    {
        gpu_kinetic_local_terms(regularized,
                                active,
                                gradient_x,
                                gradient_y,
                                gradient_z,
                                functional,
                                density_floor,
                                volume_element,
                                result.potential_hartree,
                                flux_x,
                                flux_y,
                                flux_z,
                                energy_hartree);
    }
    else
#endif
    {
#pragma omp parallel for reduction(+:energy_hartree) schedule(static)
        for (std::size_t index = 0; index < size; ++index)
        {
            const double rho = regularized[index];
            const double gradient_norm = std::sqrt(gradient_x[index] * gradient_x[index]
                                                   + gradient_y[index] * gradient_y[index]
                                                   + gradient_z[index] * gradient_z[index]);
            const double reduced_gradient
                = reduced_gradient_scale * gradient_norm / std::pow(rho, 4.0 / 3.0);
            double enhancement = 0.0;
            double enhancement_derivative = 0.0;
            kinetic_enhancement(functional,
                                reduced_gradient,
                                enhancement,
                                enhancement_derivative);
            energy_hartree
                += (c_tf * std::pow(rho, 5.0 / 3.0) * enhancement
                    - reference_energy_density)
                   * volume_element;
            result.potential_hartree[index]
                = c_tf * std::pow(rho, 2.0 / 3.0)
                  * (5.0 * enhancement / 3.0
                     - 4.0 * reduced_gradient * enhancement_derivative / 3.0);
            if (gradient_norm > 0.0)
            {
                const double flux_scale
                    = c_tf * reduced_gradient_scale * std::pow(rho, 1.0 / 3.0)
                      * enhancement_derivative / gradient_norm;
                flux_x[index] = flux_scale * gradient_x[index];
                flux_y[index] = flux_scale * gradient_y[index];
                flux_z[index] = flux_scale * gradient_z[index];
            }
        }
    }
    result.energy_hartree = energy_hartree;

    const std::vector<double> flux_divergence
        = apply_divergence(flux_x,
                           flux_y,
                           flux_z,
                           grid,
                           differential_operator);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < size; ++index)
    {
        result.potential_hartree[index]
            = active[index] ? result.potential_hartree[index] - flux_divergence[index] : 0.0;
    }
    return result;
}

ScalarFunctionalResult evaluate_unpolarized_dirac_exchange(const std::vector<double>& density,
                                                            const UniformGrid& grid,
                                                            const double density_floor,
                                                            const GridDifferentialOperator*
                                                                differential_operator)
{
    const std::size_t size = evaluation_size(grid, differential_operator);
    if (density.size() != size)
    {
        throw std::invalid_argument("FDE density does not match the local evaluation grid");
    }
    const double c_x = 0.75 * std::pow(3.0 / pi, 1.0 / 3.0);
    const double reference_energy_density = -c_x * std::pow(density_floor, 4.0 / 3.0);
    const double volume_element
        = grid.spacing_x_bohr * grid.spacing_y_bohr * grid.spacing_z_bohr;
    ScalarFunctionalResult result;
    result.energy_hartree = 0.0;
    result.potential_hartree.assign(density.size(), 0.0);
    double energy_hartree = 0.0;
#pragma omp parallel for reduction(+:energy_hartree) schedule(static)
    for (std::size_t index = 0; index < density.size(); ++index)
    {
        const double rho = std::max(density[index], density_floor);
        energy_hartree
            += (-c_x * std::pow(rho, 4.0 / 3.0) - reference_energy_density) * volume_element;
        if (density[index] > density_floor)
        {
            result.potential_hartree[index] = -4.0 * c_x * std::pow(rho, 1.0 / 3.0) / 3.0;
        }
    }
    result.energy_hartree = energy_hartree;
    return result;
}

template <typename Evaluator>
FrozenSemilocalCache prepare_frozen_functional(
    const SpinDensity& frozen_density,
    const UniformGrid& grid,
    const double density_floor,
    const GridDifferentialOperator* differential_operator,
    const Evaluator& evaluator)
{
    const std::size_t size = evaluation_size(grid, differential_operator);
    validate_density(frozen_density, size);
    if (!std::isfinite(density_floor) || density_floor <= 0.0)
    {
        throw std::invalid_argument("FDE density floor must be finite and positive");
    }

    FrozenSemilocalCache cache;
    cache.energy_ry = 0.0;
    cache.potential_ry.alpha_ry.assign(size, 0.0);
    cache.potential_ry.beta_ry.assign(size, 0.0);
    const std::vector<double>* frozen_channels[2]
        = {&frozen_density.alpha_bohr3, &frozen_density.beta_bohr3};
    std::vector<double>* cache_potentials[2]
        = {&cache.potential_ry.alpha_ry, &cache.potential_ry.beta_ry};
    const int evaluated_channels
        = equal_spin_channels(frozen_density, differential_operator) ? 1 : 2;

    for (int spin = 0; spin < evaluated_channels; ++spin)
    {
        std::vector<double> frozen_scaled(size, 0.0);
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < size; ++index)
        {
            frozen_scaled[index] = 2.0 * (*frozen_channels[spin])[index];
        }
        const ScalarFunctionalResult frozen
            = evaluator(frozen_scaled, grid, density_floor, differential_operator);
        const int multiplicity = evaluated_channels == 1 ? 2 : 1;
        cache.energy_ry += multiplicity * hartree_to_rydberg * 0.5
                           * frozen.energy_hartree;
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < size; ++index)
        {
            (*cache_potentials[spin])[index]
                = hartree_to_rydberg * frozen.potential_hartree[index];
        }
    }
    if (evaluated_channels == 1)
    {
        cache.potential_ry.beta_ry = cache.potential_ry.alpha_ry;
    }
    return cache;
}

template <typename Evaluator>
NonadditiveFunctionalResult evaluate_nonadditive_cached(
    const SpinDensity& active_density,
    const SpinDensity& frozen_density,
    const UniformGrid& grid,
    const double density_floor,
    const GridDifferentialOperator* differential_operator,
    const FrozenSemilocalCache& frozen_cache,
    const Evaluator& evaluator)
{
    const std::size_t size = evaluation_size(grid, differential_operator);
    validate_density(active_density, size);
    validate_density(frozen_density, size);
    validate_frozen_cache(frozen_cache, size);
    if (!std::isfinite(density_floor) || density_floor <= 0.0)
    {
        throw std::invalid_argument("FDE density floor must be finite and positive");
    }

    NonadditiveFunctionalResult result;
    result.energy_ry = -frozen_cache.energy_ry;
    result.active_potential.alpha_ry.assign(size, 0.0);
    result.active_potential.beta_ry.assign(size, 0.0);
    result.frozen_potential.alpha_ry.assign(size, 0.0);
    result.frozen_potential.beta_ry.assign(size, 0.0);

    const std::vector<double>* active_channels[2]
        = {&active_density.alpha_bohr3, &active_density.beta_bohr3};
    const std::vector<double>* frozen_channels[2]
        = {&frozen_density.alpha_bohr3, &frozen_density.beta_bohr3};
    const std::vector<double>* cached_frozen_potentials[2]
        = {&frozen_cache.potential_ry.alpha_ry, &frozen_cache.potential_ry.beta_ry};
    std::vector<double>* active_potentials[2]
        = {&result.active_potential.alpha_ry, &result.active_potential.beta_ry};
    std::vector<double>* frozen_potentials[2]
        = {&result.frozen_potential.alpha_ry, &result.frozen_potential.beta_ry};
    const bool active_channels_equal
        = equal_spin_channels(active_density, differential_operator);
    const bool frozen_channels_equal
        = equal_spin_channels(frozen_density, differential_operator);
    const int evaluated_channels
        = (active_channels_equal && frozen_channels_equal) ? 1 : 2;

    for (int spin = 0; spin < evaluated_channels; ++spin)
    {
        std::vector<double> active_scaled(size, 0.0);
        std::vector<double> frozen_scaled(size, 0.0);
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < size; ++index)
        {
            active_scaled[index] = 2.0 * (*active_channels[spin])[index];
            frozen_scaled[index] = 2.0 * (*frozen_channels[spin])[index];
        }
        const std::vector<double> total_scaled = add_density(active_scaled, frozen_scaled);
        const ScalarFunctionalResult total
            = evaluator(total_scaled, grid, density_floor, differential_operator);
        const ScalarFunctionalResult active
            = evaluator(active_scaled, grid, density_floor, differential_operator);
        const int multiplicity = evaluated_channels == 1 ? 2 : 1;
        result.energy_ry += multiplicity * hartree_to_rydberg * 0.5
                            * (total.energy_hartree - active.energy_hartree);
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < size; ++index)
        {
            (*active_potentials[spin])[index]
                = hartree_to_rydberg
                  * (total.potential_hartree[index] - active.potential_hartree[index]);
            (*frozen_potentials[spin])[index]
                = hartree_to_rydberg * total.potential_hartree[index]
                  - (*cached_frozen_potentials[spin])[index];
        }
    }
    if (evaluated_channels == 1)
    {
        result.active_potential.beta_ry = result.active_potential.alpha_ry;
        result.frozen_potential.beta_ry = result.frozen_potential.alpha_ry;
    }
    return result;
}

} // namespace

const char* kinetic_functional_name(const KineticFunctional functional)
{
    switch (functional)
    {
        case KineticFunctional::ThomasFermi:
            return "thomas_fermi";
        case KineticFunctional::Pw91k:
            return "pw91k";
        case KineticFunctional::RevApbek:
            return "revapbek";
    }
    throw std::invalid_argument("FDE received an unknown kinetic functional");
}

FrozenSemilocalCache SemilocalFunctional::prepare_frozen_kinetic(
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const KineticFunctional functional,
    const double density_floor_bohr3,
    const GridDifferentialOperator* differential_operator)
{
    const auto evaluator = [functional](const std::vector<double>& density,
                                        const UniformGrid& local_grid,
                                        const double floor,
                                        const GridDifferentialOperator* local_operator) {
        return evaluate_unpolarized_kinetic(density,
                                            local_grid,
                                            functional,
                                            floor,
                                            local_operator);
    };
    return prepare_frozen_functional(frozen,
                                     grid,
                                     density_floor_bohr3,
                                     differential_operator,
                                     evaluator);
}

NonadditiveFunctionalResult SemilocalFunctional::nonadditive_kinetic(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const KineticFunctional functional,
    const double density_floor_bohr3,
    const GridDifferentialOperator* differential_operator)
{
    const FrozenSemilocalCache frozen_cache
        = prepare_frozen_kinetic(frozen,
                                 grid,
                                 functional,
                                 density_floor_bohr3,
                                 differential_operator);
    return nonadditive_kinetic_cached(active,
                                      frozen,
                                      grid,
                                      functional,
                                      density_floor_bohr3,
                                      frozen_cache,
                                      differential_operator);
}

NonadditiveFunctionalResult SemilocalFunctional::nonadditive_kinetic_cached(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const KineticFunctional functional,
    const double density_floor_bohr3,
    const FrozenSemilocalCache& frozen_cache,
    const GridDifferentialOperator* differential_operator)
{
    const auto evaluator = [functional](const std::vector<double>& density,
                                        const UniformGrid& local_grid,
                                        const double floor,
                                        const GridDifferentialOperator* local_operator) {
        return evaluate_unpolarized_kinetic(density,
                                            local_grid,
                                            functional,
                                            floor,
                                            local_operator);
    };
    return evaluate_nonadditive_cached(active,
                                       frozen,
                                       grid,
                                       density_floor_bohr3,
                                       differential_operator,
                                       frozen_cache,
                                       evaluator);
}

FrozenSemilocalCache SemilocalFunctional::prepare_frozen_dirac_exchange(
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3,
    const GridDifferentialOperator* differential_operator)
{
    return prepare_frozen_functional(frozen,
                                     grid,
                                     density_floor_bohr3,
                                     differential_operator,
                                     evaluate_unpolarized_dirac_exchange);
}

NonadditiveFunctionalResult SemilocalFunctional::nonadditive_dirac_exchange(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3,
    const GridDifferentialOperator* differential_operator)
{
    const FrozenSemilocalCache frozen_cache
        = prepare_frozen_dirac_exchange(frozen,
                                        grid,
                                        density_floor_bohr3,
                                        differential_operator);
    return nonadditive_dirac_exchange_cached(active,
                                              frozen,
                                              grid,
                                              density_floor_bohr3,
                                              frozen_cache,
                                              differential_operator);
}

NonadditiveFunctionalResult SemilocalFunctional::nonadditive_dirac_exchange_cached(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3,
    const FrozenSemilocalCache& frozen_cache,
    const GridDifferentialOperator* differential_operator)
{
    return evaluate_nonadditive_cached(active,
                                       frozen,
                                       grid,
                                       density_floor_bohr3,
                                       differential_operator,
                                       frozen_cache,
                                       evaluate_unpolarized_dirac_exchange);
}

} // namespace fde
