#ifndef FDE_ONE_WAY_SCF_H
#define FDE_ONE_WAY_SCF_H

#include "source_lcao/module_fde/io/fde_density_artifact.h"
#include "source_lcao/module_fde/embedding/fde_potential_evaluator.h"
#include "source_lcao/module_fde/embedding/fde_spin_ao_matrix.h"
#include "source_lcao/module_fde/embedding/fde_subspace_solver.h"

#include <cstddef>
#include <vector>

namespace fde
{

class OneWayScfBackend
{
  public:
    virtual ~OneWayScfBackend() {}

    virtual SpinAoMatrix embedding_potential_matrix(
        const SpinPotential& real_space_potential,
        const std::size_t full_ao_dimension) const = 0;

    virtual SpinDensity density_from_ao_matrices(
        const SpinAoMatrix& density_matrices,
        const std::size_t full_ao_dimension,
        const UniformGrid& grid) const = 0;
};

struct OneWayScfControls
{
    int maximum_iterations;
    double density_tolerance;
    double electron_tolerance;
    double mixing_beta;
};

struct OneWayScfRequest
{
    FrozenDensityArtifact active_initial;
    FrozenDensityArtifact frozen;
    std::vector<double> frozen_hartree_potential_ry;
    std::vector<double> full_hamiltonian_ry;
    std::vector<double> full_overlap;
    std::size_t full_ao_dimension;
    std::vector<std::size_t> active_orbitals;
    PotFdeConfig potential_config;
    OneWayScfControls controls;
};

struct OneWayScfResult
{
    FrozenDensityArtifact active;
    ActiveSubspaceSolution alpha_solution;
    ActiveSubspaceSolution beta_solution;
    EmbeddingPotentialResult embedding;
    int iterations;
    double density_residual;
};

class OneWayScf
{
  public:
    static OneWayScfResult run(const OneWayScfRequest& request,
                               const NonadditiveXcProvider& xc_provider,
                               const OneWayScfBackend& backend);
};

} // namespace fde

#endif // FDE_ONE_WAY_SCF_H
