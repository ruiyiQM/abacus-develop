#include "source_lcao/module_fde/coupling/fde_electronic_coupling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fde
{

namespace
{

template <typename Scalar>
struct InverseDeterminant
{
    std::vector<Scalar> inverse;
    Scalar determinant;
};

template <typename Scalar>
std::vector<Scalar> occupied_overlap(const ScalarOccupiedSpinOrbitals<Scalar>& bra,
                                     const ScalarOccupiedSpinOrbitals<Scalar>& ket,
                                     const std::vector<Scalar>& ao_overlap,
                                     const std::size_t ao_dimension,
                                     const OccupiedOverlapPolicy& overlap_policy)
{
    const std::size_t bra_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(bra, ao_dimension);
    const std::size_t ket_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(ket, ao_dimension);
    if (bra_count != ket_count)
    {
        throw std::invalid_argument(
            "FDE electronic coupling requires equal spin populations in both states");
    }
    std::vector<Scalar> result(bra_count * bra_count, 0.0);
    for (std::size_t ket_orbital = 0; ket_orbital < bra_count; ++ket_orbital)
    {
        for (std::size_t bra_orbital = 0; bra_orbital < bra_count; ++bra_orbital)
        {
            if (!overlap_policy.retain(bra.source_fragment_labels[bra_orbital],
                                       ket.source_fragment_labels[ket_orbital]))
            {
                continue;
            }
            Scalar value = 0.0;
            for (std::size_t column = 0; column < ao_dimension; ++column)
            {
                Scalar overlap_times_ket = 0.0;
                for (std::size_t row = 0; row < ao_dimension; ++row)
                {
                    overlap_times_ket
                        += ao_overlap[column + row * ao_dimension]
                           * ket.coefficients[row + ket_orbital * ao_dimension];
                }
                value += conjugate_scalar(bra.coefficients[column + bra_orbital * ao_dimension])
                         * overlap_times_ket;
            }
            result[bra_orbital + ket_orbital * bra_count] = value;
        }
    }
    return result;
}

template <typename Scalar>
InverseDeterminant<Scalar> invert_and_determinant(const std::vector<Scalar>& matrix,
                                          const std::size_t dimension,
                                          const double tolerance)
{
    if (dimension == 0)
    {
        return {std::vector<Scalar>(), 1.0};
    }
    std::vector<Scalar> work(dimension * dimension, 0.0);
    std::vector<Scalar> inverse(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row)
    {
        inverse[row * dimension + row] = 1.0;
        for (std::size_t column = 0; column < dimension; ++column)
        {
            work[row * dimension + column] = matrix[row + column * dimension];
        }
    }

    Scalar determinant = 1.0;
    for (std::size_t pivot_column = 0; pivot_column < dimension; ++pivot_column)
    {
        std::size_t pivot_row = pivot_column;
        double pivot_magnitude = std::abs(work[pivot_row * dimension + pivot_column]);
        for (std::size_t row = pivot_column + 1; row < dimension; ++row)
        {
            const double candidate = std::abs(work[row * dimension + pivot_column]);
            if (candidate > pivot_magnitude)
            {
                pivot_magnitude = candidate;
                pivot_row = row;
            }
        }
        if (!std::isfinite(pivot_magnitude) || pivot_magnitude <= tolerance)
        {
            throw std::runtime_error("FDE occupied-orbital overlap matrix is singular");
        }
        if (pivot_row != pivot_column)
        {
            for (std::size_t column = 0; column < dimension; ++column)
            {
                std::swap(work[pivot_row * dimension + column],
                          work[pivot_column * dimension + column]);
                std::swap(inverse[pivot_row * dimension + column],
                          inverse[pivot_column * dimension + column]);
            }
            determinant = -determinant;
        }
        const Scalar pivot = work[pivot_column * dimension + pivot_column];
        determinant *= pivot;
        for (std::size_t column = 0; column < dimension; ++column)
        {
            work[pivot_column * dimension + column] /= pivot;
            inverse[pivot_column * dimension + column] /= pivot;
        }
        for (std::size_t row = 0; row < dimension; ++row)
        {
            if (row == pivot_column)
            {
                continue;
            }
            const Scalar factor = work[row * dimension + pivot_column];
            for (std::size_t column = 0; column < dimension; ++column)
            {
                work[row * dimension + column]
                    -= factor * work[pivot_column * dimension + column];
                inverse[row * dimension + column]
                    -= factor * inverse[pivot_column * dimension + column];
            }
        }
    }

    std::vector<Scalar> inverse_column_major(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row)
    {
        for (std::size_t column = 0; column < dimension; ++column)
        {
            inverse_column_major[row + column * dimension]
                = inverse[row * dimension + column];
        }
    }
    return {inverse_column_major, determinant};
}

template <typename Scalar>
std::vector<Scalar> transition_density(const ScalarOccupiedSpinOrbitals<Scalar>& bra,
                                       const ScalarOccupiedSpinOrbitals<Scalar>& ket,
                                       const std::vector<Scalar>& inverse_overlap,
                                       const std::size_t ao_dimension,
                                       const OccupiedOverlapPolicy& overlap_policy)
{
    const std::size_t count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(bra, ao_dimension);
    std::vector<Scalar> density(ao_dimension * ao_dimension, 0.0);
    for (std::size_t column = 0; column < ao_dimension; ++column)
    {
        for (std::size_t row = 0; row < ao_dimension; ++row)
        {
            Scalar value = 0.0;
            for (std::size_t ket_orbital = 0; ket_orbital < count; ++ket_orbital)
            {
                for (std::size_t bra_orbital = 0; bra_orbital < count; ++bra_orbital)
                {
                    // The density contraction must use the same retained blocks as the overlap.
                    if (!overlap_policy.retain(bra.source_fragment_labels[bra_orbital],
                                               ket.source_fragment_labels[ket_orbital]))
                    {
                        continue;
                    }
                    value += ket.coefficients[row + ket_orbital * ao_dimension]
                             * inverse_overlap[ket_orbital
                                               + bra_orbital * count]
                             * conjugate_scalar(bra.coefficients[column + bra_orbital * ao_dimension]);
                }
            }
            density[row + column * ao_dimension] = value;
        }
    }
    return density;
}

template <typename Scalar>
double raw_self_overlap(const ScalarDiabaticDeterminantArtifact<Scalar>& artifact,
                        const std::vector<Scalar>& ao_overlap,
                        const OccupiedOverlapPolicy& overlap_policy,
                        const double tolerance)
{
    const std::size_t dimension = artifact.ao_dimension;
    const std::size_t alpha_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(artifact.alpha, dimension);
    const std::size_t beta_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(artifact.beta, dimension);
    const auto alpha_overlap = occupied_overlap(
        artifact.alpha, artifact.alpha, ao_overlap, dimension, overlap_policy);
    const auto beta_overlap = occupied_overlap(
        artifact.beta, artifact.beta, ao_overlap, dimension, overlap_policy);
    const auto alpha = invert_and_determinant(alpha_overlap, alpha_count, tolerance);
    const auto beta = invert_and_determinant(beta_overlap, beta_count, tolerance);
    const Scalar result = alpha.determinant * beta.determinant;
    const double norm = std::real(result);
    const double imaginary_norm = std::imag(result);
    const double imaginary_error = std::abs(imaginary_norm);
    if (!std::isfinite(norm) || !std::isfinite(imaginary_error)
        || imaginary_error > tolerance || norm <= tolerance)
    {
        throw std::runtime_error("FDE diabatic determinant has a singular, complex or negative norm");
    }
    return norm;
}

template <typename Scalar>
void validate_pair_metadata(const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
                            const ScalarDiabaticDeterminantArtifact<Scalar>& ket)
{
    if (bra.ao_dimension != ket.ao_dimension
        || bra.geometry_fingerprint != ket.geometry_fingerprint
        || bra.orbital_fingerprint != ket.orbital_fingerprint
        || bra.boundary_twist != ket.boundary_twist
        || bra.resource_fingerprint != ket.resource_fingerprint)
    {
        throw std::invalid_argument(
            "FDE electronic coupling determinant metadata is incompatible");
    }
}

} // namespace

bool FullOccupiedOverlapPolicy::retain(const std::string& bra_fragment,
                                       const std::string& ket_fragment) const
{
    (void)bra_fragment;
    (void)ket_fragment;
    return true;
}

template <typename Scalar>
ScalarDeterminantTransition<Scalar> ScalarElectronicCoupling<Scalar>::transition(
    const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
    const ScalarDiabaticDeterminantArtifact<Scalar>& ket,
    const std::vector<Scalar>& ao_overlap,
    const double singular_value_tolerance)
{
    const FullOccupiedOverlapPolicy full_overlap;
    return ScalarElectronicCoupling<Scalar>::transition_with_policy(bra,
                                                      ket,
                                                      ao_overlap,
                                                      full_overlap,
                                                      singular_value_tolerance);
}

template <typename Scalar>
ScalarDeterminantTransition<Scalar> ScalarElectronicCoupling<Scalar>::transition_with_policy(
    const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
    const ScalarDiabaticDeterminantArtifact<Scalar>& ket,
    const std::vector<Scalar>& ao_overlap,
    const OccupiedOverlapPolicy& overlap_policy,
    const double singular_value_tolerance)
{
    if (!std::isfinite(singular_value_tolerance) || singular_value_tolerance <= 0.0)
    {
        throw std::invalid_argument("FDE electronic-coupling singular tolerance is invalid");
    }
    validate_pair_metadata(bra, ket);
    ScalarDeterminantArtifactIO<Scalar>::validate(bra, ao_overlap, 1.0e-8);
    ScalarDeterminantArtifactIO<Scalar>::validate(ket, ao_overlap, 1.0e-8);

    const std::size_t dimension = bra.ao_dimension;
    const std::size_t alpha_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(bra.alpha, dimension);
    const std::size_t beta_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(bra.beta, dimension);
    const auto alpha_overlap = occupied_overlap(
        bra.alpha, ket.alpha, ao_overlap, dimension, overlap_policy);
    const auto beta_overlap = occupied_overlap(
        bra.beta, ket.beta, ao_overlap, dimension, overlap_policy);
    const auto alpha = invert_and_determinant(alpha_overlap, alpha_count, singular_value_tolerance);
    const auto beta = invert_and_determinant(beta_overlap, beta_count, singular_value_tolerance);

    ScalarDeterminantTransition<Scalar> result;
    result.raw_overlap = alpha.determinant * beta.determinant;
    const double bra_norm = raw_self_overlap(bra,
                                             ao_overlap,
                                             overlap_policy,
                                             singular_value_tolerance);
    const double ket_norm = raw_self_overlap(ket,
                                             ao_overlap,
                                             overlap_policy,
                                             singular_value_tolerance);
    const double norm_product = bra_norm * ket_norm;
    const double normalization = std::sqrt(norm_product);
    result.normalized_overlap = result.raw_overlap / normalization;
    const double overlap_magnitude = std::abs(result.normalized_overlap);
    if (!std::isfinite(overlap_magnitude) || overlap_magnitude > 1.0 + 1.0e-8)
    {
        throw std::runtime_error("FDE normalized determinant overlap is invalid");
    }
    result.density_matrix.alpha
        = transition_density(bra.alpha, ket.alpha, alpha.inverse, dimension, overlap_policy);
    result.density_matrix.beta
        = transition_density(bra.beta, ket.beta, beta.inverse, dimension, overlap_policy);
    return result;
}

template <typename Scalar>
ScalarElectronicCouplingResult<Scalar> ScalarElectronicCoupling<Scalar>::evaluate_symmetric(
    const ScalarDiabaticDeterminantArtifact<Scalar>& first,
    const ScalarDiabaticDeterminantArtifact<Scalar>& second,
    const std::vector<Scalar>& ao_overlap,
    const ScalarTransitionEnergyProvider<Scalar>& energy_provider,
    const CouplingValidationControls& validation,
    const double singular_value_tolerance)
{
    const FullOccupiedOverlapPolicy full_overlap;
    return ScalarElectronicCoupling<Scalar>::evaluate_symmetric_with_policy(first,
                                                              second,
                                                              ao_overlap,
                                                              full_overlap,
                                                              energy_provider,
                                                              validation,
                                                              singular_value_tolerance);
}

template <typename Scalar>
ScalarElectronicCouplingResult<Scalar> ScalarElectronicCoupling<Scalar>::evaluate_symmetric_with_policy(
    const ScalarDiabaticDeterminantArtifact<Scalar>& first,
    const ScalarDiabaticDeterminantArtifact<Scalar>& second,
    const std::vector<Scalar>& ao_overlap,
    const OccupiedOverlapPolicy& overlap_policy,
    const ScalarTransitionEnergyProvider<Scalar>& energy_provider,
    const CouplingValidationControls& validation,
    const double singular_value_tolerance)
{
    if (!std::isfinite(validation.overlap_reciprocity_tolerance)
        || validation.overlap_reciprocity_tolerance < 0.0
        || !std::isfinite(validation.transition_density_trace_tolerance)
        || validation.transition_density_trace_tolerance <= 0.0)
    {
        throw std::invalid_argument(
            "FDE coupling validation tolerances are invalid");
    }
    const std::string provider_name = energy_provider.name();
    if (provider_name.empty()
        || provider_name.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(
            "FDE transition-energy provider name must be a nonempty token");
    }
    const ScalarDeterminantTransition<Scalar> forward
        = ScalarElectronicCoupling<Scalar>::transition_with_policy(first,
                                                     second,
                                                     ao_overlap,
                                                     overlap_policy,
                                                     singular_value_tolerance);
    const ScalarDeterminantTransition<Scalar> reverse
        = ScalarElectronicCoupling<Scalar>::transition_with_policy(second,
                                                     first,
                                                     ao_overlap,
                                                     overlap_policy,
                                                     singular_value_tolerance);
    const Scalar reverse_overlap = conjugate_scalar(reverse.normalized_overlap);
    const Scalar overlap_difference = forward.normalized_overlap - reverse_overlap;
    const double overlap_reciprocity_error = std::abs(overlap_difference);
    if (overlap_reciprocity_error > validation.overlap_reciprocity_tolerance)
    {
        throw std::runtime_error("FDE forward and reverse determinant overlaps disagree");
    }

    const std::size_t alpha_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(first.alpha,
                                                first.ao_dimension);
    const std::size_t beta_count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(first.beta,
                                                first.ao_dimension);
    double maximum_trace_error = 0.0;
    const ScalarSpinTransitionDensityMatrix<Scalar>* densities[2]
        = {&forward.density_matrix, &reverse.density_matrix};
    for (std::size_t direction = 0; direction < 2; ++direction)
    {
        const Scalar alpha_trace = ScalarElectronicCoupling<Scalar>::electron_count(
            densities[direction]->alpha, ao_overlap, first.ao_dimension);
        const Scalar beta_trace = ScalarElectronicCoupling<Scalar>::electron_count(
            densities[direction]->beta, ao_overlap, first.ao_dimension);
        const Scalar alpha_difference = alpha_trace - static_cast<double>(alpha_count);
        const Scalar beta_difference = beta_trace - static_cast<double>(beta_count);
        const double alpha_error = std::abs(alpha_difference);
        const double beta_error = std::abs(beta_difference);
        maximum_trace_error = std::max(maximum_trace_error, alpha_error);
        maximum_trace_error = std::max(maximum_trace_error, beta_error);
    }

    if (maximum_trace_error > validation.transition_density_trace_tolerance)
    {
        throw std::runtime_error(
            "FDE transition-density electron trace exceeds its tolerance");
    }

    ScalarElectronicCouplingResult<Scalar> result;
    result.provider = provider_name;
    result.normalized_overlap
        = 0.5 * (forward.normalized_overlap + reverse_overlap);
    result.overlap_reciprocity_error = overlap_reciprocity_error;
    result.maximum_transition_density_trace_error = maximum_trace_error;
    result.forward_transition_energy_ry
        = energy_provider.evaluate_ry(first,
                                      second,
                                      forward.density_matrix,
                                      first.ao_dimension);
    result.reverse_transition_energy_ry
        = energy_provider.evaluate_ry(second,
                                      first,
                                      reverse.density_matrix,
                                      first.ao_dimension);
    const double forward_magnitude = std::abs(result.forward_transition_energy_ry);
    const double reverse_magnitude = std::abs(result.reverse_transition_energy_ry);
    if (!std::isfinite(forward_magnitude) || !std::isfinite(reverse_magnitude))
    {
        throw std::runtime_error("FDE transition-energy provider returned a non-finite value");
    }
    const Scalar conjugate_reverse_energy = conjugate_scalar(result.reverse_transition_energy_ry);
    result.hamiltonian_coupling_ry
        = 0.5 * result.normalized_overlap
          * (result.forward_transition_energy_ry
             + conjugate_reverse_energy);
    const Scalar energy_difference = result.forward_transition_energy_ry - conjugate_reverse_energy;
    result.transition_energy_asymmetry_ry = std::abs(energy_difference);
    result.estimated_coupling_uncertainty_ry
        = 0.5 * std::abs(result.normalized_overlap)
          * result.transition_energy_asymmetry_ry;
    result.forward_density_matrix = forward.density_matrix;
    result.reverse_density_matrix = reverse.density_matrix;
    return result;
}

template <typename Scalar>
Scalar ScalarElectronicCoupling<Scalar>::electron_count(const std::vector<Scalar>& transition_density,
                                          const std::vector<Scalar>& ao_overlap,
                                          const std::size_t ao_dimension)
{
    if (transition_density.size() != ao_dimension * ao_dimension
        || ao_overlap.size() != ao_dimension * ao_dimension)
    {
        throw std::invalid_argument("FDE transition-density trace dimensions are inconsistent");
    }
    Scalar result = 0.0;
    for (std::size_t column = 0; column < ao_dimension; ++column)
    {
        for (std::size_t row = 0; row < ao_dimension; ++row)
        {
            result += transition_density[row + column * ao_dimension]
                      * ao_overlap[column + row * ao_dimension];
        }
    }
    return result;
}

template <typename Scalar>
ScalarOrthogonalizedPair<Scalar> ScalarElectronicCoupling<Scalar>::orthogonalize_pair(
    const double first_energy_ry,
    const double second_energy_ry,
    const Scalar overlap,
    const Scalar hamiltonian_coupling_ry,
    const double overlap_eigenvalue_cutoff)
{
    const double overlap_magnitude = std::abs(overlap);
    const double hamiltonian_magnitude = std::abs(hamiltonian_coupling_ry);
    if (!std::isfinite(first_energy_ry) || !std::isfinite(second_energy_ry)
        || !std::isfinite(overlap_magnitude) || !std::isfinite(hamiltonian_magnitude)
        || !std::isfinite(overlap_eigenvalue_cutoff)
        || overlap_eigenvalue_cutoff <= 0.0 || overlap_eigenvalue_cutoff >= 1.0)
    {
        throw std::invalid_argument("FDE two-state orthogonalization inputs are invalid");
    }
    const double smallest_eigenvalue = 1.0 - overlap_magnitude;
    if (smallest_eigenvalue <= overlap_eigenvalue_cutoff)
    {
        throw std::runtime_error("FDE two-state overlap is singular at the eigenvalue cutoff");
    }
    const double largest_eigenvalue = 1.0 + overlap_magnitude;
    const double positive_root = std::sqrt(largest_eigenvalue);
    const double negative_root = std::sqrt(smallest_eigenvalue);
    const double root_product = positive_root * negative_root;
    const double root_sum = positive_root + negative_root;
    const double diagonal = root_sum / (2.0 * root_product);
    // This form avoids subtracting nearly equal roots at the Gamma/zero-overlap limit.
    const double off_diagonal_scale = -1.0 / (root_product * root_sum);
    const Scalar conjugate_overlap = conjugate_scalar(overlap);
    const Scalar conjugate_hamiltonian = conjugate_scalar(hamiltonian_coupling_ry);
    const std::vector<Scalar> inverse_root = {diagonal,
                                              off_diagonal_scale * conjugate_overlap,
                                              off_diagonal_scale * overlap,
                                              diagonal};
    const std::vector<Scalar> hamiltonian = {first_energy_ry,
                                             conjugate_hamiltonian,
                                             hamiltonian_coupling_ry,
                                             second_energy_ry};
    ScalarOrthogonalizedPair<Scalar> result;
    result.hamiltonian_ry.assign(4, 0.0);
    for (std::size_t column = 0; column < 2; ++column)
    {
        for (std::size_t row = 0; row < 2; ++row)
        {
            for (std::size_t left = 0; left < 2; ++left)
            {
                for (std::size_t right = 0; right < 2; ++right)
                {
                    result.hamiltonian_ry[row + column * 2]
                        += inverse_root[row + left * 2]
                           * hamiltonian[left + right * 2]
                           * inverse_root[right + column * 2];
                }
            }
            const double magnitude = std::abs(result.hamiltonian_ry[row + column * 2]);
            if (!std::isfinite(magnitude))
            {
                throw std::runtime_error("FDE orthogonalized Hamiltonian is non-finite");
            }
        }
    }
    result.coupling_ry = result.hamiltonian_ry[2];
    const double first_diagonal = std::real(result.hamiltonian_ry[0]);
    const double second_diagonal = std::real(result.hamiltonian_ry[3]);
    const double midpoint = 0.5 * (first_diagonal + second_diagonal);
    const double half_difference = 0.5 * (first_diagonal - second_diagonal);
    const double coupling_magnitude = std::abs(result.coupling_ry);
    const double gap = std::hypot(half_difference, coupling_magnitude);
    result.eigenvalues_ry = {midpoint - gap, midpoint + gap};
    return result;
}

template class ScalarElectronicCoupling<double>;
template class ScalarElectronicCoupling<std::complex<double>>;

} // namespace fde
