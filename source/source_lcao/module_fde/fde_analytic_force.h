#ifndef FDE_ANALYTIC_FORCE_H
#define FDE_ANALYTIC_FORCE_H

#include "fde_potential_evaluator.h"
#include "fde_spin_ao_matrix.h"

#include <cstddef>
#include <string>
#include <vector>

namespace fde
{

/**
 * Backend contraction of one local spin potential with one subsystem density
 * matrix. The returned Cartesian array is an ABACUS-sign force (-dE/dR) in
 * atom-major xyz order and Ry/Bohr.
 */
class LocalPotentialForceBackend
{
  public:
    virtual ~LocalPotentialForceBackend() {}

    virtual std::vector<double> force_from_local_potential(
        const SpinPotential& potential,
        const SpinAoMatrix& density_matrix,
        std::size_t full_ao_dimension,
        std::size_t atom_count) const = 0;
};

struct FdeForceFragment
{
    std::string label;
    SpinDensity density;
    std::vector<double> hartree_potential_ry;
    SpinAoMatrix density_matrix;
};

struct DiagonalFdeForceRequest
{
    std::vector<FdeForceFragment> fragments;
    std::size_t full_ao_dimension;
    std::size_t atom_count;
    PotFdeConfig potential_config;
    std::vector<double> base_force_ry_per_bohr;
    double hartree_reciprocity_tolerance_ry;
};

struct DiagonalFdeForceResult
{
    std::vector<double> base_force_ry_per_bohr;
    std::vector<double> hartree_cross_force_ry_per_bohr;
    std::vector<double> nonadditive_kinetic_force_ry_per_bohr;
    std::vector<double> nonadditive_xc_force_ry_per_bohr;
    std::vector<double> total_force_ry_per_bohr;
    double hartree_cross_energy_ry;
    double nonadditive_kinetic_energy_ry;
    double nonadditive_xc_energy_ry;
};

/**
 * Analytic semilocal FDE correction for one converged diagonal diabatic state.
 *
 * The base force owns all ordinary ABACUS terms exactly once and excludes the
 * three density-dependent embedding corrections returned here. Each correction
 * is contracted for every fragment density. A deterministic telescoping sum
 * evaluates F[rho_total] - sum_I F[rho_I], so the same API supports two or
 * more fragments without treating the environment as one artificial fragment.
 */
class DiagonalFdeAnalyticForce
{
  public:
    static DiagonalFdeForceResult evaluate(
        const DiagonalFdeForceRequest& request,
        const NonadditiveXcProvider& xc_provider,
        const LocalPotentialForceBackend& backend);
};

} // namespace fde

#endif // FDE_ANALYTIC_FORCE_H
