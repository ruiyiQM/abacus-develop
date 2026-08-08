#include "fde_electronic_coupling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fde
{

namespace
{

struct InverseDeterminant
{
    std::vector<double> inverse;
    double determinant;
};

std::vector<double> occupied_overlap(const OccupiedSpinOrbitals& bra,
                                     const OccupiedSpinOrbitals& ket,
                                     const std::vector<double>& ao_overlap,
                                     const std::size_t ao_dimension,
                                     const OccupiedOverlapPolicy& overlap_policy)
{
    const std::size_t bra_count
        = DeterminantArtifactIO::occupied_count(bra, ao_dimension);
    const std::size_t ket_count
        = DeterminantArtifactIO::occupied_count(ket, ao_dimension);
    if (bra_count != ket_count)
    {
        throw std::invalid_argument(
            "FDE electronic coupling requires equal spin populations in both states");
    }
    std::vector<double> result(bra_count * bra_count, 0.0);
    for (std::size_t ket_orbital = 0; ket_orbital < bra_count; ++ket_orbital)
    {
        for (std::size_t bra_orbital = 0; bra_orbital < bra_count; ++bra_orbital)
        {
            if (!overlap_policy.retain(bra.source_fragment_labels[bra_orbital],
                                       ket.source_fragment_labels[ket_orbital]))
            {
                continue;
            }
            double value = 0.0;
            for (std::size_t column = 0; column < ao_dimension; ++column)
            {
                double overlap_times_ket = 0.0;
                for (std::size_t row = 0; row < ao_dimension; ++row)
                {
                    overlap_times_ket
                        += ao_overlap[column + row * ao_dimension]
                           * ket.coefficients[row + ket_orbital * ao_dimension];
                }
                value += bra.coefficients[column + bra_orbital * ao_dimension]
                         * overlap_times_ket;
            }
            result[bra_orbital + ket_orbital * bra_count] = value;
        }
    }
    return result;
}

InverseDeterminant invert_and_determinant(const std::vector<double>& matrix,
                                          const std::size_t dimension,
                                          const double tolerance)
{
    if (dimension == 0)
    {
        return {std::vector<double>(), 1.0};
    }
    std::vector<double> work(dimension * dimension, 0.0);
    std::vector<double> inverse(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row)
    {
        inverse[row * dimension + row] = 1.0;
        for (std::size_t column = 0; column < dimension; ++column)
        {
            work[row * dimension + column] = matrix[row + column * dimension];
        }
    }

    double determinant = 1.0;
    for (std::size_t pivot_column = 0; pivot_column < dimension; ++pivot_column)
    {
        std::size_t pivot_row = pivot_column;
        double pivot_magnitude = std::fabs(work[pivot_row * dimension + pivot_column]);
        for (std::size_t row = pivot_column + 1; row < dimension; ++row)
        {
            const double candidate = std::fabs(work[row * dimension + pivot_column]);
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
        const double pivot = work[pivot_column * dimension + pivot_column];
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
            const double factor = work[row * dimension + pivot_column];
            for (std::size_t column = 0; column < dimension; ++column)
            {
                work[row * dimension + column]
                    -= factor * work[pivot_column * dimension + column];
                inverse[row * dimension + column]
                    -= factor * inverse[pivot_column * dimension + column];
            }
        }
    }

    std::vector<double> inverse_column_major(dimension * dimension, 0.0);
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

std::vector<double> transition_density(const OccupiedSpinOrbitals& bra,
                                       const OccupiedSpinOrbitals& ket,
                                       const std::vector<double>& inverse_overlap,
                                       const std::size_t ao_dimension)
{
    const std::size_t count
        = DeterminantArtifactIO::occupied_count(bra, ao_dimension);
    std::vector<double> density(ao_dimension * ao_dimension, 0.0);
    for (std::size_t column = 0; column < ao_dimension; ++column)
    {
        for (std::size_t row = 0; row < ao_dimension; ++row)
        {
            double value = 0.0;
            for (std::size_t ket_orbital = 0; ket_orbital < count; ++ket_orbital)
            {
                for (std::size_t bra_orbital = 0; bra_orbital < count; ++bra_orbital)
                {
                    value += ket.coefficients[row + ket_orbital * ao_dimension]
                             * inverse_overlap[ket_orbital
                                               + bra_orbital * count]
                             * bra.coefficients[column + bra_orbital * ao_dimension];
                }
            }
            density[row + column * ao_dimension] = value;
        }
    }
    return density;
}

double raw_self_overlap(const DiabaticDeterminantArtifact& artifact,
                        const std::vector<double>& ao_overlap,
                        const OccupiedOverlapPolicy& overlap_policy,
                        const double tolerance)
{
    const std::size_t dimension = artifact.ao_dimension;
    const std::size_t alpha_count
        = DeterminantArtifactIO::occupied_count(artifact.alpha, dimension);
    const std::size_t beta_count
        = DeterminantArtifactIO::occupied_count(artifact.beta, dimension);
    const double alpha = invert_and_determinant(
                             occupied_overlap(artifact.alpha,
                                              artifact.alpha,
                                              ao_overlap,
                                              dimension,
                                              overlap_policy),
                             alpha_count,
                             tolerance)
                             .determinant;
    const double beta = invert_and_determinant(
                            occupied_overlap(artifact.beta,
                                             artifact.beta,
                                             ao_overlap,
                                             dimension,
                                             overlap_policy),
                            beta_count,
                            tolerance)
                            .determinant;
    const double result = alpha * beta;
    if (!std::isfinite(result) || result <= tolerance)
    {
        throw std::runtime_error("FDE diabatic determinant has a singular or negative norm");
    }
    return result;
}

void validate_pair_metadata(const DiabaticDeterminantArtifact& bra,
                            const DiabaticDeterminantArtifact& ket)
{
    if (bra.ao_dimension != ket.ao_dimension
        || bra.geometry_fingerprint != ket.geometry_fingerprint
        || bra.orbital_fingerprint != ket.orbital_fingerprint)
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

DeterminantTransition ElectronicCoupling::transition(
    const DiabaticDeterminantArtifact& bra,
    const DiabaticDeterminantArtifact& ket,
    const std::vector<double>& ao_overlap,
    const double singular_value_tolerance)
{
    const FullOccupiedOverlapPolicy full_overlap;
    return ElectronicCoupling::transition_with_policy(bra,
                                                      ket,
                                                      ao_overlap,
                                                      full_overlap,
                                                      singular_value_tolerance);
}

DeterminantTransition ElectronicCoupling::transition_with_policy(
    const DiabaticDeterminantArtifact& bra,
    const DiabaticDeterminantArtifact& ket,
    const std::vector<double>& ao_overlap,
    const OccupiedOverlapPolicy& overlap_policy,
    const double singular_value_tolerance)
{
    if (!std::isfinite(singular_value_tolerance) || singular_value_tolerance <= 0.0)
    {
        throw std::invalid_argument("FDE electronic-coupling singular tolerance is invalid");
    }
    validate_pair_metadata(bra, ket);
    DeterminantArtifactIO::validate(bra, ao_overlap, 1.0e-8);
    DeterminantArtifactIO::validate(ket, ao_overlap, 1.0e-8);

    const std::size_t dimension = bra.ao_dimension;
    const std::size_t alpha_count
        = DeterminantArtifactIO::occupied_count(bra.alpha, dimension);
    const std::size_t beta_count
        = DeterminantArtifactIO::occupied_count(bra.beta, dimension);
    const InverseDeterminant alpha = invert_and_determinant(
        occupied_overlap(bra.alpha,
                         ket.alpha,
                         ao_overlap,
                         dimension,
                         overlap_policy),
        alpha_count,
        singular_value_tolerance);
    const InverseDeterminant beta = invert_and_determinant(
        occupied_overlap(bra.beta,
                         ket.beta,
                         ao_overlap,
                         dimension,
                         overlap_policy),
        beta_count,
        singular_value_tolerance);

    DeterminantTransition result;
    result.raw_overlap = alpha.determinant * beta.determinant;
    const double bra_norm = raw_self_overlap(bra,
                                             ao_overlap,
                                             overlap_policy,
                                             singular_value_tolerance);
    const double ket_norm = raw_self_overlap(ket,
                                             ao_overlap,
                                             overlap_policy,
                                             singular_value_tolerance);
    result.normalized_overlap = result.raw_overlap / std::sqrt(bra_norm * ket_norm);
    if (!std::isfinite(result.normalized_overlap)
        || std::fabs(result.normalized_overlap) > 1.0 + 1.0e-8)
    {
        throw std::runtime_error("FDE normalized determinant overlap is invalid");
    }
    result.density_matrix.alpha
        = transition_density(bra.alpha, ket.alpha, alpha.inverse, dimension);
    result.density_matrix.beta
        = transition_density(bra.beta, ket.beta, beta.inverse, dimension);
    return result;
}

ElectronicCouplingResult ElectronicCoupling::evaluate_symmetric(
    const DiabaticDeterminantArtifact& first,
    const DiabaticDeterminantArtifact& second,
    const std::vector<double>& ao_overlap,
    const TransitionEnergyProvider& energy_provider,
    const double singular_value_tolerance)
{
    const FullOccupiedOverlapPolicy full_overlap;
    return ElectronicCoupling::evaluate_symmetric_with_policy(first,
                                                              second,
                                                              ao_overlap,
                                                              full_overlap,
                                                              energy_provider,
                                                              singular_value_tolerance);
}

ElectronicCouplingResult ElectronicCoupling::evaluate_symmetric_with_policy(
    const DiabaticDeterminantArtifact& first,
    const DiabaticDeterminantArtifact& second,
    const std::vector<double>& ao_overlap,
    const OccupiedOverlapPolicy& overlap_policy,
    const TransitionEnergyProvider& energy_provider,
    const double singular_value_tolerance)
{
    const DeterminantTransition forward
        = ElectronicCoupling::transition_with_policy(first,
                                                     second,
                                                     ao_overlap,
                                                     overlap_policy,
                                                     singular_value_tolerance);
    const DeterminantTransition reverse
        = ElectronicCoupling::transition_with_policy(second,
                                                     first,
                                                     ao_overlap,
                                                     overlap_policy,
                                                     singular_value_tolerance);
    if (std::fabs(forward.normalized_overlap - reverse.normalized_overlap) > 1.0e-10)
    {
        throw std::runtime_error("FDE forward and reverse determinant overlaps disagree");
    }

    ElectronicCouplingResult result;
    result.normalized_overlap
        = 0.5 * (forward.normalized_overlap + reverse.normalized_overlap);
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
    if (!std::isfinite(result.forward_transition_energy_ry)
        || !std::isfinite(result.reverse_transition_energy_ry))
    {
        throw std::runtime_error("FDE transition-energy provider returned a non-finite value");
    }
    result.hamiltonian_coupling_ry
        = 0.5 * result.normalized_overlap
          * (result.forward_transition_energy_ry
             + result.reverse_transition_energy_ry);
    result.forward_density_matrix = forward.density_matrix;
    result.reverse_density_matrix = reverse.density_matrix;
    return result;
}

double ElectronicCoupling::electron_count(const std::vector<double>& transition_density,
                                          const std::vector<double>& ao_overlap,
                                          const std::size_t ao_dimension)
{
    if (transition_density.size() != ao_dimension * ao_dimension
        || ao_overlap.size() != ao_dimension * ao_dimension)
    {
        throw std::invalid_argument("FDE transition-density trace dimensions are inconsistent");
    }
    double result = 0.0;
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

} // namespace fde
