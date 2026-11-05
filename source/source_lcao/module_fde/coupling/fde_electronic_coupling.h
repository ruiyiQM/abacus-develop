#ifndef FDE_ELECTRONIC_COUPLING_H
#define FDE_ELECTRONIC_COUPLING_H

#include "source_lcao/module_fde/io/fde_determinant_artifact.h"

#include <cstddef>
#include <string>
#include <vector>

namespace fde
{

template <typename Scalar>
struct ScalarSpinTransitionDensityMatrix
{
    std::vector<Scalar> alpha;
    std::vector<Scalar> beta;
};

template <typename Scalar>
struct ScalarDeterminantTransition
{
    Scalar raw_overlap;
    Scalar normalized_overlap;
    ScalarSpinTransitionDensityMatrix<Scalar> density_matrix;
};

class OccupiedOverlapPolicy
{
  public:
    virtual ~OccupiedOverlapPolicy() {}

    virtual bool retain(const std::string& bra_fragment,
                        const std::string& ket_fragment) const = 0;
};

class FullOccupiedOverlapPolicy : public OccupiedOverlapPolicy
{
  public:
    bool retain(const std::string& bra_fragment,
                const std::string& ket_fragment) const override;
};

template <typename Scalar>
class ScalarTransitionEnergyProvider
{
  public:
    virtual ~ScalarTransitionEnergyProvider() {}

    virtual std::string name() const = 0;

    virtual Scalar evaluate_ry(const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
                               const ScalarDiabaticDeterminantArtifact<Scalar>& ket,
                               const ScalarSpinTransitionDensityMatrix<Scalar>& transition_density,
                               std::size_t ao_dimension) const = 0;
};

struct CouplingValidationControls
{
    double overlap_reciprocity_tolerance;
    double transition_density_trace_tolerance;
};

template <typename Scalar>
struct ScalarElectronicCouplingResult
{
    Scalar normalized_overlap;
    Scalar forward_transition_energy_ry;
    Scalar reverse_transition_energy_ry;
    Scalar hamiltonian_coupling_ry;
    double overlap_reciprocity_error;
    double maximum_transition_density_trace_error;
    double transition_energy_asymmetry_ry;
    double estimated_coupling_uncertainty_ry;
    std::string provider;
    ScalarSpinTransitionDensityMatrix<Scalar> forward_density_matrix;
    ScalarSpinTransitionDensityMatrix<Scalar> reverse_density_matrix;
};

template <typename Scalar>
struct ScalarOrthogonalizedPair
{
    std::vector<Scalar> hamiltonian_ry;
    std::vector<double> eigenvalues_ry;
    std::vector<Scalar> coefficients;
    Scalar coupling_ry;
    double maximum_residual;
    double maximum_metric_error;
};

template <typename Scalar>
class ScalarElectronicCoupling
{
  public:
    static ScalarDeterminantTransition<Scalar> transition(
        const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
        const ScalarDiabaticDeterminantArtifact<Scalar>& ket,
        const std::vector<Scalar>& ao_overlap,
        double singular_value_tolerance);

    static ScalarDeterminantTransition<Scalar> transition_with_policy(
        const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
        const ScalarDiabaticDeterminantArtifact<Scalar>& ket,
        const std::vector<Scalar>& ao_overlap,
        const OccupiedOverlapPolicy& overlap_policy,
        double singular_value_tolerance);

    static ScalarElectronicCouplingResult<Scalar> evaluate_symmetric(
        const ScalarDiabaticDeterminantArtifact<Scalar>& first,
        const ScalarDiabaticDeterminantArtifact<Scalar>& second,
        const std::vector<Scalar>& ao_overlap,
        const ScalarTransitionEnergyProvider<Scalar>& energy_provider,
        const CouplingValidationControls& validation,
        double singular_value_tolerance);

    static ScalarElectronicCouplingResult<Scalar> evaluate_symmetric_with_policy(
        const ScalarDiabaticDeterminantArtifact<Scalar>& first,
        const ScalarDiabaticDeterminantArtifact<Scalar>& second,
        const std::vector<Scalar>& ao_overlap,
        const OccupiedOverlapPolicy& overlap_policy,
        const ScalarTransitionEnergyProvider<Scalar>& energy_provider,
        const CouplingValidationControls& validation,
        double singular_value_tolerance);

    static Scalar electron_count(const std::vector<Scalar>& transition_density,
                                 const std::vector<Scalar>& ao_overlap,
                                 std::size_t ao_dimension);

    static ScalarOrthogonalizedPair<Scalar> orthogonalize_pair(
        double first_energy_ry,
        double second_energy_ry,
        Scalar overlap,
        Scalar hamiltonian_coupling_ry,
        double overlap_eigenvalue_cutoff);
};

using SpinTransitionDensityMatrix = ScalarSpinTransitionDensityMatrix<double>;
using ComplexSpinTransitionDensityMatrix = ScalarSpinTransitionDensityMatrix<std::complex<double>>;
using DeterminantTransition = ScalarDeterminantTransition<double>;
using ComplexDeterminantTransition = ScalarDeterminantTransition<std::complex<double>>;
using TransitionEnergyProvider = ScalarTransitionEnergyProvider<double>;
using ComplexTransitionEnergyProvider = ScalarTransitionEnergyProvider<std::complex<double>>;
using ElectronicCouplingResult = ScalarElectronicCouplingResult<double>;
using ComplexElectronicCouplingResult = ScalarElectronicCouplingResult<std::complex<double>>;
using ElectronicCoupling = ScalarElectronicCoupling<double>;
using ComplexElectronicCoupling = ScalarElectronicCoupling<std::complex<double>>;

} // namespace fde

#endif // FDE_ELECTRONIC_COUPLING_H
