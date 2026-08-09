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

/** Derivatives on the local part of a real-space grid. */
class GridDifferentialOperator
{
  public:
    virtual ~GridDifferentialOperator() {}

    virtual std::size_t local_size() const = 0;
    virtual void gradient(const std::vector<double>& values,
                          std::vector<double>& gradient_x,
                          std::vector<double>& gradient_y,
                          std::vector<double>& gradient_z) const = 0;
    virtual std::vector<double> divergence(const std::vector<double>& vector_x,
                                           const std::vector<double>& vector_y,
                                           const std::vector<double>& vector_z) const = 0;
};

class SemilocalFunctional
{
  public:
    static NonadditiveFunctionalResult nonadditive_kinetic(
        const SpinDensity& active,
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const KineticFunctional functional,
        const double density_floor_bohr3,
        const GridDifferentialOperator* differential_operator = nullptr);

    static NonadditiveFunctionalResult nonadditive_dirac_exchange(
        const SpinDensity& active,
        const SpinDensity& frozen,
        const UniformGrid& grid,
        const double density_floor_bohr3,
        const GridDifferentialOperator* differential_operator = nullptr);
};

} // namespace fde

#endif // FDE_SEMILOCAL_FUNCTIONAL_H
