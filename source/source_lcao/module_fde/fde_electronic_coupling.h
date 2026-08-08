#ifndef FDE_ELECTRONIC_COUPLING_H
#define FDE_ELECTRONIC_COUPLING_H

#include "fde_determinant_artifact.h"

#include <cstddef>
#include <string>
#include <vector>

namespace fde
{

struct SpinTransitionDensityMatrix
{
    std::vector<double> alpha;
    std::vector<double> beta;
};

struct DeterminantTransition
{
    double raw_overlap;
    double normalized_overlap;
    SpinTransitionDensityMatrix density_matrix;
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

class TransitionEnergyProvider
{
  public:
    virtual ~TransitionEnergyProvider() {}

    virtual double evaluate_ry(const DiabaticDeterminantArtifact& bra,
                               const DiabaticDeterminantArtifact& ket,
                               const SpinTransitionDensityMatrix& transition_density,
                               std::size_t ao_dimension) const = 0;
};

struct ElectronicCouplingResult
{
    double normalized_overlap;
    double forward_transition_energy_ry;
    double reverse_transition_energy_ry;
    double hamiltonian_coupling_ry;
    SpinTransitionDensityMatrix forward_density_matrix;
    SpinTransitionDensityMatrix reverse_density_matrix;
};

class ElectronicCoupling
{
  public:
    static DeterminantTransition transition(
        const DiabaticDeterminantArtifact& bra,
        const DiabaticDeterminantArtifact& ket,
        const std::vector<double>& ao_overlap,
        double singular_value_tolerance);

    static DeterminantTransition transition_with_policy(
        const DiabaticDeterminantArtifact& bra,
        const DiabaticDeterminantArtifact& ket,
        const std::vector<double>& ao_overlap,
        const OccupiedOverlapPolicy& overlap_policy,
        double singular_value_tolerance);

    static ElectronicCouplingResult evaluate_symmetric(
        const DiabaticDeterminantArtifact& first,
        const DiabaticDeterminantArtifact& second,
        const std::vector<double>& ao_overlap,
        const TransitionEnergyProvider& energy_provider,
        double singular_value_tolerance);

    static ElectronicCouplingResult evaluate_symmetric_with_policy(
        const DiabaticDeterminantArtifact& first,
        const DiabaticDeterminantArtifact& second,
        const std::vector<double>& ao_overlap,
        const OccupiedOverlapPolicy& overlap_policy,
        const TransitionEnergyProvider& energy_provider,
        double singular_value_tolerance);

    static double electron_count(const std::vector<double>& transition_density,
                                 const std::vector<double>& ao_overlap,
                                 std::size_t ao_dimension);
};

} // namespace fde

#endif // FDE_ELECTRONIC_COUPLING_H
