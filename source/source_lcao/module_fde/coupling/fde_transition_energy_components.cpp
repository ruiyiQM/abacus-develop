#include "source_lcao/module_fde/coupling/fde_transition_energy_components.h"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace fde
{
namespace
{

template <typename Scalar>
bool finite_value(const Scalar value)
{
    const double real = std::real(value);
    const double imaginary = std::imag(value);
    return std::isfinite(real) && std::isfinite(imaginary);
}

} // namespace

template <typename Scalar>
Scalar one_body_energy_ry(const std::vector<Scalar>& density,
                          const std::vector<Scalar>& hamiltonian_ry,
                          const std::size_t dimension)
{
    if (dimension == 0 || dimension > std::numeric_limits<std::size_t>::max() / dimension)
    {
        throw std::invalid_argument("FDE one-body AO dimension is invalid or overflows");
    }
    const std::size_t matrix_size = dimension * dimension;
    if (density.size() != matrix_size || hamiltonian_ry.size() != matrix_size)
    {
        throw std::invalid_argument("FDE one-body contraction dimensions are inconsistent");
    }
    Scalar energy = 0.0;
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            const Scalar rho = density[row + column * dimension];
            const Scalar h = hamiltonian_ry[column + row * dimension];
            if (!finite_value(rho) || !finite_value(h))
            {
                throw std::invalid_argument("FDE one-body matrices must be finite");
            }
            energy += rho * h;
        }
    }
    if (!finite_value(energy))
    {
        throw std::overflow_error("FDE one-body energy is not finite");
    }
    return energy;
}

template double one_body_energy_ry(const std::vector<double>&,
                                   const std::vector<double>&,
                                   std::size_t);
template std::complex<double> one_body_energy_ry(const std::vector<std::complex<double>>&,
                                                const std::vector<std::complex<double>>&,
                                                std::size_t);

} // namespace fde
