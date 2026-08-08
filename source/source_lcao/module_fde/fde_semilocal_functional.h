#ifndef FDE_SEMILOCAL_FUNCTIONAL_H
#define FDE_SEMILOCAL_FUNCTIONAL_H

#include <cstddef>
#include <vector>

namespace fde
{

enum class KineticFunctional
{
    ThomasFermi,
    Lc94Pw91k
};

struct UniformGrid
{
    std::size_t x;
    std::size_t y;
    std::size_t z;
    double spacing_x_bohr;
    double spacing_y_bohr;
    double spacing_z_bohr;
};

struct SpinDensity
{
    std::vector<double> alpha_bohr3;
    std::vector<double> beta_bohr3;
};

struct SpinPotential
{
    std::vector<double> alpha_ry;
    std::vector<double> beta_ry;
};

struct NonadditiveFunctionalResult
{
    double energy_ry;
    SpinPotential active_potential;
    SpinPotential frozen_potential;
};

class SemilocalFunctional
{
  public:
    static NonadditiveFunctionalResult nonadditive_kinetic(
        const SpinDensity& active,
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const KineticFunctional functional,
        const double density_floor_bohr3);

    static NonadditiveFunctionalResult nonadditive_dirac_exchange(
        const SpinDensity& active,
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3);
};

} // namespace fde

#endif // FDE_SEMILOCAL_FUNCTIONAL_H
