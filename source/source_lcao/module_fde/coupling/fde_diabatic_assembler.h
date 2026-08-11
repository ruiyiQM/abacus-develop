#ifndef FDE_DIABATIC_ASSEMBLER_H
#define FDE_DIABATIC_ASSEMBLER_H

#include "source_lcao/module_fde/coupling/fde_electronic_coupling.h"
#include "source_lcao/module_fde/coupling/fde_multistate_solver.h"

#include <cstddef>
#include <string>
#include <vector>

namespace fde
{

struct FdeDiabApproximationSpec
{
    std::vector<std::size_t> state_indices;
    std::vector<std::string> overlap_active_fragments;
    std::vector<std::string> determinant_fragments;
};

struct FdeDiabApproximationAudit
{
    std::size_t k_states;
    std::size_t l_overlap_active_fragments;
    std::size_t m_determinant_fragments;
    std::vector<std::size_t> omitted_state_indices;
    std::vector<std::string> omitted_determinant_fragments;
    std::size_t retained_interfragment_overlap_pairs;
    std::size_t discarded_interfragment_overlap_pairs;
};

struct DiabaticPairAssembly
{
    std::size_t first_input_state;
    std::size_t second_input_state;
    double normalized_overlap;
    double hamiltonian_coupling_ry;
    double overlap_reciprocity_error;
    double maximum_transition_density_trace_error;
    double transition_energy_asymmetry_ry;
    double estimated_coupling_uncertainty_ry;
    std::string provider;
};

struct FdeDiabaticAssemblyResult
{
    NonorthogonalStateProblem problem;
    FdeDiabApproximationAudit audit;
    std::vector<DiabaticPairAssembly> pairs;
};

class FdeDiabFragmentOverlapPolicy : public OccupiedOverlapPolicy
{
  public:
    explicit FdeDiabFragmentOverlapPolicy(
        const std::vector<std::string>& overlap_active_fragments);

    bool retain(const std::string& bra_fragment,
                const std::string& ket_fragment) const override;

  private:
    std::vector<std::string> overlap_active_fragments_;
};

class FdeDiabaticAssembler
{
  public:
    static FdeDiabaticAssemblyResult assemble(
        const std::vector<DiabaticDeterminantArtifact>& determinants,
        const std::vector<double>& diagonal_energies_ry,
        const std::vector<double>& ao_overlap,
        const FdeDiabApproximationSpec& approximation,
        const TransitionEnergyProvider& transition_energy_provider,
        const CouplingValidationControls& validation,
        double singular_value_tolerance);
};

} // namespace fde

#endif // FDE_DIABATIC_ASSEMBLER_H
