#include "fde_one_way_scf.h"

#include <cmath>
#include <stdexcept>

namespace fde
{

namespace
{

std::vector<double> occupations(const int electrons, const std::size_t dimension)
{
    if (electrons < 0 || static_cast<std::size_t>(electrons) > dimension)
    {
        throw std::invalid_argument("FDE subsystem spin population exceeds the active AO dimension");
    }
    std::vector<double> values(dimension, 0.0);
    for (int index = 0; index < electrons; ++index)
    {
        values[static_cast<std::size_t>(index)] = 1.0;
    }
    return values;
}

void validate_controls(const OneWayScfControls& controls)
{
    if (controls.maximum_iterations <= 0
        || !std::isfinite(controls.density_tolerance) || controls.density_tolerance <= 0.0
        || !std::isfinite(controls.electron_tolerance) || controls.electron_tolerance < 0.0
        || !std::isfinite(controls.mixing_beta) || controls.mixing_beta <= 0.0
        || controls.mixing_beta > 1.0)
    {
        throw std::invalid_argument("FDE one-way SCF controls are invalid");
    }
}

SpinDensity artifact_density(const FrozenDensityArtifact& artifact)
{
    return {artifact.rho_alpha_bohr3, artifact.rho_beta_bohr3};
}

void validate_density(const SpinDensity& density,
                      const FrozenDensityArtifact& artifact,
                      const UniformGrid& grid,
                      const double electron_tolerance)
{
    const std::size_t size = grid.x * grid.y * grid.z;
    if (density.alpha_bohr3.size() != size || density.beta_bohr3.size() != size)
    {
        throw std::invalid_argument("FDE backend returned a density with the wrong grid size");
    }
    const double volume_element
        = grid.spacing_x_bohr * grid.spacing_y_bohr * grid.spacing_z_bohr;
    double alpha_electrons = 0.0;
    double beta_electrons = 0.0;
    for (std::size_t index = 0; index < size; ++index)
    {
        if (!std::isfinite(density.alpha_bohr3[index]) || density.alpha_bohr3[index] < 0.0
            || !std::isfinite(density.beta_bohr3[index]) || density.beta_bohr3[index] < 0.0)
        {
            throw std::invalid_argument("FDE backend returned a negative or non-finite density");
        }
        alpha_electrons += density.alpha_bohr3[index] * volume_element;
        beta_electrons += density.beta_bohr3[index] * volume_element;
    }
    if (std::fabs(alpha_electrons - artifact.alpha_electrons) > electron_tolerance
        || std::fabs(beta_electrons - artifact.beta_electrons) > electron_tolerance)
    {
        throw std::invalid_argument("FDE backend density does not preserve fixed spin populations");
    }
}

double density_residual(const SpinDensity& first,
                        const SpinDensity& second,
                        const double volume_element)
{
    double squared = 0.0;
    for (std::size_t index = 0; index < first.alpha_bohr3.size(); ++index)
    {
        const double alpha_difference = first.alpha_bohr3[index] - second.alpha_bohr3[index];
        const double beta_difference = first.beta_bohr3[index] - second.beta_bohr3[index];
        squared += (alpha_difference * alpha_difference + beta_difference * beta_difference)
                   * volume_element;
    }
    return std::sqrt(squared);
}

SpinDensity mix_density(const SpinDensity& old_density,
                        const SpinDensity& new_density,
                        const double beta)
{
    SpinDensity mixed = old_density;
    for (std::size_t index = 0; index < old_density.alpha_bohr3.size(); ++index)
    {
        mixed.alpha_bohr3[index]
            = (1.0 - beta) * old_density.alpha_bohr3[index]
              + beta * new_density.alpha_bohr3[index];
        mixed.beta_bohr3[index]
            = (1.0 - beta) * old_density.beta_bohr3[index]
              + beta * new_density.beta_bohr3[index];
    }
    return mixed;
}

std::vector<double> add_matrix(const std::vector<double>& first,
                               const std::vector<double>& second)
{
    if (first.size() != second.size())
    {
        throw std::invalid_argument("FDE AO potential matrix size does not match the base Hamiltonian");
    }
    std::vector<double> sum(first.size(), 0.0);
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        sum[index] = first[index] + second[index];
    }
    return sum;
}

} // namespace

OneWayScfResult OneWayScf::run(const OneWayScfRequest& request,
                              const NonadditiveXcProvider& xc_provider,
                              const OneWayScfBackend& backend)
{
    validate_controls(request.controls);
    DensityArtifactIO::validate_compatible_pair(request.active_initial,
                                                request.frozen,
                                                request.controls.electron_tolerance);
    const std::size_t grid_points = request.potential_config.grid.x
                                    * request.potential_config.grid.y
                                    * request.potential_config.grid.z;
    if (request.frozen_hartree_potential_ry.size() != grid_points)
    {
        throw std::invalid_argument("FDE frozen Hartree potential does not match the density grid");
    }

    const std::vector<double> alpha_occupations
        = occupations(request.active_initial.alpha_electrons, request.active_orbitals.size());
    const std::vector<double> beta_occupations
        = occupations(request.active_initial.beta_electrons, request.active_orbitals.size());
    SpinDensity active_density = artifact_density(request.active_initial);
    const SpinDensity frozen_density = artifact_density(request.frozen);
    const double volume_element = request.potential_config.grid.spacing_x_bohr
                                  * request.potential_config.grid.spacing_y_bohr
                                  * request.potential_config.grid.spacing_z_bohr;

    OneWayScfResult result;
    result.active = request.active_initial;
    result.iterations = 0;
    result.density_residual = 0.0;
    result.active.scf_converged = false;
    for (int iteration = 1; iteration <= request.controls.maximum_iterations; ++iteration)
    {
        result.embedding = EmbeddingPotentialEvaluator::evaluate(
            active_density,
            frozen_density,
            request.frozen_hartree_potential_ry,
            request.potential_config,
            xc_provider);
        const SpinAoMatrix embedding_matrices
            = backend.embedding_potential_matrix(result.embedding.potential,
                                                 request.full_ao_dimension);
        const std::vector<double> alpha_hamiltonian
            = add_matrix(request.full_hamiltonian_ry, embedding_matrices.alpha);
        const std::vector<double> beta_hamiltonian
            = add_matrix(request.full_hamiltonian_ry, embedding_matrices.beta);
        const ActiveSubspaceProblem alpha_problem
            = SubspaceSolver::build_problem(alpha_hamiltonian,
                                            request.full_overlap,
                                            request.full_ao_dimension,
                                            request.active_orbitals,
                                            true,
                                            1.0e-10);
        const ActiveSubspaceProblem beta_problem
            = SubspaceSolver::build_problem(beta_hamiltonian,
                                            request.full_overlap,
                                            request.full_ao_dimension,
                                            request.active_orbitals,
                                            true,
                                            1.0e-10);
        result.alpha_solution = SubspaceSolver::solve_serial(alpha_problem, alpha_occupations);
        result.beta_solution = SubspaceSolver::solve_serial(beta_problem, beta_occupations);
        const SpinAoMatrix density_matrices{result.alpha_solution.full_density_matrix,
                                            result.beta_solution.full_density_matrix};
        const SpinDensity new_density
            = backend.density_from_ao_matrices(density_matrices,
                                              request.full_ao_dimension,
                                              request.potential_config.grid);
        validate_density(new_density,
                         request.active_initial,
                         request.potential_config.grid,
                         request.controls.electron_tolerance);
        result.density_residual = density_residual(new_density, active_density, volume_element);
        result.iterations = iteration;
        if (result.density_residual <= request.controls.density_tolerance)
        {
            active_density = new_density;
            result.active.scf_converged = true;
            break;
        }
        active_density = mix_density(active_density, new_density, request.controls.mixing_beta);
    }

    result.active.rho_alpha_bohr3 = active_density.alpha_bohr3;
    result.active.rho_beta_bohr3 = active_density.beta_bohr3;
    result.active.freeze_thaw_cycle = request.active_initial.freeze_thaw_cycle + 1;
    DensityArtifactIO::validate(result.active, request.controls.electron_tolerance);
    return result;
}

} // namespace fde
