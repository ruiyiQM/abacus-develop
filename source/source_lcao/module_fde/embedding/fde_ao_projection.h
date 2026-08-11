#ifndef FDE_EMBEDDING_AO_PROJECTION_H
#define FDE_EMBEDDING_AO_PROJECTION_H

#include <cstddef>
#include <vector>

namespace fde
{

/**
 * Select the active-fragment principal submatrix from a full supersystem
 * matrix. Matrices use the ABACUS column-major convention.
 *
 * The full matrix must be assembled before this projection. In particular,
 * local and nonlocal pseudopotential operators from environment atoms must
 * already be present in the supplied matrix.
 */
class ActiveAoProjection
{
  public:
    static void validate(const std::size_t full_dimension,
                         const std::vector<std::size_t>& active_orbitals);

    static std::vector<double> principal_submatrix(
        const std::vector<double>& full_matrix,
        const std::size_t full_dimension,
        const std::vector<std::size_t>& active_orbitals);

    static double quadratic_form(const std::vector<double>& matrix,
                                 const std::size_t dimension,
                                 const std::vector<double>& coefficients);
};

} // namespace fde

#endif // FDE_EMBEDDING_AO_PROJECTION_H
