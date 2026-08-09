#ifndef FDE_DIABATIC_POSTPROCESS_H
#define FDE_DIABATIC_POSTPROCESS_H

#include "fde_diabatic_assembler.h"
#include "fde_linearized_state.h"
#include "fde_runtime_config.h"

#include <cstddef>
#include <string>
#include <vector>

namespace fde
{

struct OrthogonalizedPairCoupling
{
    std::size_t first_state;
    std::size_t second_state;
    double coupling_ry;
};

struct DiabaticPostprocessResult
{
    FdeDiabaticAssemblyResult assembly;
    NonorthogonalStateSolution adiabatic_solution;
    std::vector<OrthogonalizedPairCoupling> orthogonalized_pair_couplings;
};

class DiabaticPostprocessor
{
  public:
    static DiabaticPostprocessResult evaluate(
        const FdeRuntimeConfig& config,
        const std::vector<DiabaticDeterminantArtifact>& determinants,
        const std::vector<LinearizedStateArtifact>& linearized_states,
        const std::vector<double>& ao_overlap);

    /** Read the sidecar and artifacts, evaluate, and write prefix outputs. */
    static DiabaticPostprocessResult run(const std::string& config_path);
};

} // namespace fde

#endif // FDE_DIABATIC_POSTPROCESS_H
