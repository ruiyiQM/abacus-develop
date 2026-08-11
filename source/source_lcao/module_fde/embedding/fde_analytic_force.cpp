#include "source_lcao/module_fde/embedding/fde_analytic_force.h"

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fde
{

namespace
{

void validate_finite_vector(const std::vector<double>& values,
                            const std::size_t expected_size,
                            const char* description)
{
    if (values.size() != expected_size)
    {
        throw std::invalid_argument(std::string("FDE analytic force ") + description
                                    + " has an invalid size");
    }
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (!std::isfinite(values[index]))
        {
            throw std::invalid_argument(std::string("FDE analytic force ") + description
                                        + " contains a non-finite value");
        }
    }
}

void validate_density_matrix(const SpinAoMatrix& density_matrix,
                             const std::size_t ao_dimension)
{
    const std::size_t matrix_size = ao_dimension * ao_dimension;
    validate_finite_vector(density_matrix.alpha, matrix_size, "alpha AO density matrix");
    validate_finite_vector(density_matrix.beta, matrix_size, "beta AO density matrix");
}

SpinPotential spin_independent_potential(const std::vector<double>& potential_ry)
{
    return {potential_ry, potential_ry};
}

std::vector<double> add_pair_contractions(
    const SpinPotential& aggregate_potential,
    const SpinPotential& fragment_potential,
    const SpinAoMatrix& aggregate_density_matrix,
    const SpinAoMatrix& fragment_density_matrix,
    const DiagonalFdeForceRequest& request,
    const LocalPotentialForceBackend& backend)
{
    std::vector<double> result
        = backend.force_from_local_potential(aggregate_potential,
                                             aggregate_density_matrix,
                                             request.full_ao_dimension,
                                             request.atom_count);
    const std::vector<double> fragment
        = backend.force_from_local_potential(fragment_potential,
                                             fragment_density_matrix,
                                             request.full_ao_dimension,
                                             request.atom_count);
    const std::size_t force_size = 3 * request.atom_count;
    validate_finite_vector(result, force_size, "aggregate local-potential contribution");
    validate_finite_vector(fragment, force_size, "fragment local-potential contribution");
    for (std::size_t coordinate = 0; coordinate < force_size; ++coordinate)
    {
        result[coordinate] += fragment[coordinate];
    }
    return result;
}

void validate_functional_result(const NonadditiveFunctionalResult& result,
                                const std::size_t grid_size,
                                const char* description)
{
    if (!std::isfinite(result.energy_ry))
    {
        throw std::invalid_argument(std::string("FDE analytic force ") + description
                                    + " energy is non-finite");
    }
    validate_finite_vector(result.active_potential.alpha_ry,
                           grid_size,
                           "aggregate alpha functional potential");
    validate_finite_vector(result.active_potential.beta_ry,
                           grid_size,
                           "aggregate beta functional potential");
    validate_finite_vector(result.frozen_potential.alpha_ry,
                           grid_size,
                           "fragment alpha functional potential");
    validate_finite_vector(result.frozen_potential.beta_ry,
                           grid_size,
                           "fragment beta functional potential");
}

void add_force(std::vector<double>& destination, const std::vector<double>& contribution)
{
    for (std::size_t coordinate = 0; coordinate < destination.size(); ++coordinate)
    {
        destination[coordinate] += contribution[coordinate];
    }
}

void add_density(SpinDensity& destination, const SpinDensity& contribution)
{
    for (std::size_t index = 0; index < destination.alpha_bohr3.size(); ++index)
    {
        destination.alpha_bohr3[index] += contribution.alpha_bohr3[index];
        destination.beta_bohr3[index] += contribution.beta_bohr3[index];
    }
}

void add_density_matrix(SpinAoMatrix& destination, const SpinAoMatrix& contribution)
{
    for (std::size_t index = 0; index < destination.alpha.size(); ++index)
    {
        destination.alpha[index] += contribution.alpha[index];
        destination.beta[index] += contribution.beta[index];
    }
}

void add_scalar_field(std::vector<double>& destination,
                      const std::vector<double>& contribution)
{
    for (std::size_t index = 0; index < destination.size(); ++index)
    {
        destination[index] += contribution[index];
    }
}

double cross_hartree_energy(const SpinDensity& density,
                            const std::vector<double>& other_hartree_potential_ry,
                            const double volume_element)
{
    double energy = 0.0;
    for (std::size_t index = 0; index < other_hartree_potential_ry.size(); ++index)
    {
        energy += (density.alpha_bohr3[index] + density.beta_bohr3[index])
                  * other_hartree_potential_ry[index] * volume_element;
    }
    return energy;
}

} // namespace

DiagonalFdeForceResult DiagonalFdeAnalyticForce::evaluate(
    const DiagonalFdeForceRequest& request,
    const NonadditiveXcProvider& xc_provider,
    const LocalPotentialForceBackend& backend)
{
    if (request.fragments.size() < 2 || request.atom_count == 0
        || request.full_ao_dimension == 0
        || !std::isfinite(request.hartree_reciprocity_tolerance_ry)
        || request.hartree_reciprocity_tolerance_ry < 0.0)
    {
        throw std::invalid_argument("FDE analytic force dimensions or tolerance are invalid");
    }
    const std::size_t grid_size = request.potential_config.grid.x
                                  * request.potential_config.grid.y
                                  * request.potential_config.grid.z;
    const std::size_t force_size = 3 * request.atom_count;
    validate_finite_vector(request.base_force_ry_per_bohr,
                           force_size,
                           "base force");
    std::set<std::string> labels;
    for (std::size_t fragment = 0; fragment < request.fragments.size(); ++fragment)
    {
        const FdeForceFragment& input = request.fragments[fragment];
        if (input.label.empty()
            || input.label.find_first_of(" \t\r\n") != std::string::npos
            || !labels.insert(input.label).second)
        {
            throw std::invalid_argument(
                "FDE analytic force fragment labels must be unique nonempty tokens");
        }
        validate_finite_vector(input.hartree_potential_ry,
                               grid_size,
                               "fragment Hartree potential");
        validate_density_matrix(input.density_matrix, request.full_ao_dimension);
    }

    const double volume_element
        = request.potential_config.grid.spacing_x_bohr
          * request.potential_config.grid.spacing_y_bohr
          * request.potential_config.grid.spacing_z_bohr;

    DiagonalFdeForceResult result;
    result.base_force_ry_per_bohr = request.base_force_ry_per_bohr;
    result.hartree_cross_force_ry_per_bohr.assign(force_size, 0.0);
    result.nonadditive_kinetic_force_ry_per_bohr.assign(force_size, 0.0);
    result.nonadditive_xc_force_ry_per_bohr.assign(force_size, 0.0);
    result.hartree_cross_energy_ry = 0.0;
    result.nonadditive_kinetic_energy_ry = 0.0;
    result.nonadditive_xc_energy_ry = 0.0;

    SpinDensity aggregate_density = request.fragments[0].density;
    SpinAoMatrix aggregate_density_matrix = request.fragments[0].density_matrix;
    std::vector<double> aggregate_hartree
        = request.fragments[0].hartree_potential_ry;
    for (std::size_t fragment = 1; fragment < request.fragments.size(); ++fragment)
    {
        const FdeForceFragment& input = request.fragments[fragment];
        const NonadditiveFunctionalResult kinetic
            = SemilocalFunctional::nonadditive_kinetic(
                aggregate_density,
                input.density,
                request.potential_config.grid,
                request.potential_config.kinetic_functional,
                request.potential_config.density_floor_bohr3);
        const NonadditiveFunctionalResult xc
            = xc_provider.evaluate(aggregate_density,
                                   input.density,
                                   request.potential_config.grid,
                                   request.potential_config.density_floor_bohr3);
        validate_functional_result(kinetic, grid_size, "nonadditive kinetic");
        validate_functional_result(xc, grid_size, "nonadditive XC");

        const double aggregate_fragment_hartree
            = cross_hartree_energy(aggregate_density,
                                   input.hartree_potential_ry,
                                   volume_element);
        const double fragment_aggregate_hartree
            = cross_hartree_energy(input.density,
                                   aggregate_hartree,
                                   volume_element);
        if (std::abs(aggregate_fragment_hartree - fragment_aggregate_hartree)
            > request.hartree_reciprocity_tolerance_ry)
        {
            throw std::invalid_argument(
                "FDE analytic force requires reciprocal fragment Hartree potentials");
        }

        add_force(
            result.hartree_cross_force_ry_per_bohr,
            add_pair_contractions(
                spin_independent_potential(input.hartree_potential_ry),
                spin_independent_potential(aggregate_hartree),
                aggregate_density_matrix,
                input.density_matrix,
                request,
                backend));
        add_force(result.nonadditive_kinetic_force_ry_per_bohr,
                  add_pair_contractions(kinetic.active_potential,
                                        kinetic.frozen_potential,
                                        aggregate_density_matrix,
                                        input.density_matrix,
                                        request,
                                        backend));
        add_force(result.nonadditive_xc_force_ry_per_bohr,
                  add_pair_contractions(xc.active_potential,
                                        xc.frozen_potential,
                                        aggregate_density_matrix,
                                        input.density_matrix,
                                        request,
                                        backend));
        result.hartree_cross_energy_ry
            += 0.5 * (aggregate_fragment_hartree
                      + fragment_aggregate_hartree);
        result.nonadditive_kinetic_energy_ry += kinetic.energy_ry;
        result.nonadditive_xc_energy_ry += xc.energy_ry;

        add_density(aggregate_density, input.density);
        add_density_matrix(aggregate_density_matrix, input.density_matrix);
        add_scalar_field(aggregate_hartree, input.hartree_potential_ry);
    }
    result.total_force_ry_per_bohr = result.base_force_ry_per_bohr;
    for (std::size_t coordinate = 0; coordinate < force_size; ++coordinate)
    {
        result.total_force_ry_per_bohr[coordinate]
            += result.hartree_cross_force_ry_per_bohr[coordinate]
               + result.nonadditive_kinetic_force_ry_per_bohr[coordinate]
               + result.nonadditive_xc_force_ry_per_bohr[coordinate];
    }
    validate_finite_vector(result.total_force_ry_per_bohr,
                           force_size,
                           "total force");
    return result;
}

} // namespace fde
