#include "fde_semilocal_functional.h"

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

std::size_t offset(const std::size_t x,
                   const std::size_t y,
                   const std::size_t z,
                   const UniformGrid& grid)
{
    return x + grid.x * (y + grid.y * z);
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

void gradient(const std::vector<double>& values,
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
    for (std::size_t z = 0; z < grid.z; ++z)
    {
        for (std::size_t y = 0; y < grid.y; ++y)
        {
            for (std::size_t x = 0; x < grid.x; ++x)
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

std::vector<double> divergence(const std::vector<double>& vector_x,
                               const std::vector<double>& vector_y,
                               const std::vector<double>& vector_z,
                               const UniformGrid& grid)
{
    std::vector<double> result(vector_x.size(), 0.0);
#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t z = 0; z < grid.z; ++z)
    {
        for (std::size_t y = 0; y < grid.y; ++y)
        {
            for (std::size_t x = 0; x < grid.x; ++x)
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
                                                     const double density_floor)
{
    const std::size_t size = grid_size(grid);
    std::vector<double> regularized(size, density_floor);
    std::vector<unsigned char> active(size, 0);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < size; ++index)
    {
        regularized[index] = std::max(density[index], density_floor);
        active[index] = density[index] > density_floor ? 1 : 0;
    }

    std::vector<double> gradient_x;
    std::vector<double> gradient_y;
    std::vector<double> gradient_z;
    gradient(regularized, grid, gradient_x, gradient_y, gradient_z);

    const double c_tf = 0.3 * std::pow(3.0 * pi * pi, 2.0 / 3.0);
    const double reduced_gradient_scale = 1.0 / (2.0 * std::pow(3.0 * pi * pi, 1.0 / 3.0));
    const double reference_energy_density = c_tf * std::pow(density_floor, 5.0 / 3.0);
    const double volume_element
        = grid.spacing_x_bohr * grid.spacing_y_bohr * grid.spacing_z_bohr;
    ScalarFunctionalResult result;
    result.energy_hartree = 0.0;
    result.potential_hartree.assign(size, 0.0);
    std::vector<double> flux_x(size, 0.0);
    std::vector<double> flux_y(size, 0.0);
    std::vector<double> flux_z(size, 0.0);

    double energy_hartree = 0.0;
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
            += (c_tf * std::pow(rho, 5.0 / 3.0) * enhancement - reference_energy_density)
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
    result.energy_hartree = energy_hartree;

    const std::vector<double> flux_divergence = divergence(flux_x, flux_y, flux_z, grid);
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
                                                            const double density_floor)
{
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
NonadditiveFunctionalResult evaluate_nonadditive(const SpinDensity& active_density,
                                                 const SpinDensity& frozen_density,
                                                 const UniformGrid& grid,
                                                 const double density_floor,
                                                 const Evaluator& evaluator)
{
    const std::size_t size = grid_size(grid);
    validate_density(active_density, size);
    validate_density(frozen_density, size);
    if (!std::isfinite(density_floor) || density_floor <= 0.0)
    {
        throw std::invalid_argument("FDE density floor must be finite and positive");
    }

    NonadditiveFunctionalResult result;
    result.energy_ry = 0.0;
    result.active_potential.alpha_ry.assign(size, 0.0);
    result.active_potential.beta_ry.assign(size, 0.0);
    result.frozen_potential.alpha_ry.assign(size, 0.0);
    result.frozen_potential.beta_ry.assign(size, 0.0);

    const std::vector<double>* active_channels[2]
        = {&active_density.alpha_bohr3, &active_density.beta_bohr3};
    const std::vector<double>* frozen_channels[2]
        = {&frozen_density.alpha_bohr3, &frozen_density.beta_bohr3};
    std::vector<double>* active_potentials[2]
        = {&result.active_potential.alpha_ry, &result.active_potential.beta_ry};
    std::vector<double>* frozen_potentials[2]
        = {&result.frozen_potential.alpha_ry, &result.frozen_potential.beta_ry};

    for (int spin = 0; spin < 2; ++spin)
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
        const ScalarFunctionalResult total = evaluator(total_scaled, grid, density_floor);
        const ScalarFunctionalResult active = evaluator(active_scaled, grid, density_floor);
        const ScalarFunctionalResult frozen = evaluator(frozen_scaled, grid, density_floor);
        result.energy_ry += hartree_to_rydberg * 0.5
                            * (total.energy_hartree - active.energy_hartree
                               - frozen.energy_hartree);
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < size; ++index)
        {
            (*active_potentials[spin])[index]
                = hartree_to_rydberg
                  * (total.potential_hartree[index] - active.potential_hartree[index]);
            (*frozen_potentials[spin])[index]
                = hartree_to_rydberg
                  * (total.potential_hartree[index] - frozen.potential_hartree[index]);
        }
    }
    return result;
}

} // namespace

NonadditiveFunctionalResult SemilocalFunctional::nonadditive_kinetic(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const KineticFunctional functional,
    const double density_floor_bohr3)
{
    const auto evaluator = [functional](const std::vector<double>& density,
                                        const UniformGrid& local_grid,
                                        const double floor) {
        return evaluate_unpolarized_kinetic(density, local_grid, functional, floor);
    };
    return evaluate_nonadditive(active, frozen, grid, density_floor_bohr3, evaluator);
}

NonadditiveFunctionalResult SemilocalFunctional::nonadditive_dirac_exchange(
    const SpinDensity& active,
    const SpinDensity& frozen,
    const UniformGrid& grid,
    const double density_floor_bohr3)
{
    return evaluate_nonadditive(active,
                                frozen,
                                grid,
                                density_floor_bohr3,
                                evaluate_unpolarized_dirac_exchange);
}

} // namespace fde
