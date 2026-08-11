#include "source_lcao/module_fde/embedding/fde_ao_projection.h"

#include <stdexcept>

namespace fde
{

void ActiveAoProjection::validate(const std::size_t full_dimension,
                                  const std::vector<std::size_t>& active_orbitals)
{
    if (full_dimension == 0)
    {
        throw std::invalid_argument("FDE supersystem AO dimension must be positive");
    }
    if (active_orbitals.empty())
    {
        throw std::invalid_argument("FDE active AO set must not be empty");
    }

    for (std::size_t index = 0; index < active_orbitals.size(); ++index)
    {
        if (active_orbitals[index] >= full_dimension)
        {
            throw std::out_of_range("FDE active AO index is outside the supersystem basis");
        }
        if (index > 0 && active_orbitals[index - 1] >= active_orbitals[index])
        {
            throw std::invalid_argument("FDE active AO indices must be strictly increasing");
        }
    }
}

std::vector<double> ActiveAoProjection::principal_submatrix(
    const std::vector<double>& full_matrix,
    const std::size_t full_dimension,
    const std::vector<std::size_t>& active_orbitals)
{
    ActiveAoProjection::validate(full_dimension, active_orbitals);
    if (full_matrix.size() != full_dimension * full_dimension)
    {
        throw std::invalid_argument("FDE full AO matrix has an inconsistent size");
    }

    const std::size_t active_dimension = active_orbitals.size();
    std::vector<double> projected(active_dimension * active_dimension, 0.0);
    for (std::size_t column = 0; column < active_dimension; ++column)
    {
        for (std::size_t row = 0; row < active_dimension; ++row)
        {
            const std::size_t full_row = active_orbitals[row];
            const std::size_t full_column = active_orbitals[column];
            projected[row + column * active_dimension]
                = full_matrix[full_row + full_column * full_dimension];
        }
    }
    return projected;
}

double ActiveAoProjection::quadratic_form(const std::vector<double>& matrix,
                                          const std::size_t dimension,
                                          const std::vector<double>& coefficients)
{
    if (matrix.size() != dimension * dimension || coefficients.size() != dimension)
    {
        throw std::invalid_argument("FDE quadratic-form dimensions are inconsistent");
    }

    double value = 0.0;
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            value += coefficients[row] * matrix[row + column * dimension]
                     * coefficients[column];
        }
    }
    return value;
}

} // namespace fde
