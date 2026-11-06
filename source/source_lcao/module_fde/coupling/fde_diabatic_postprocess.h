#ifndef FDE_DIABATIC_POSTPROCESS_H
#define FDE_DIABATIC_POSTPROCESS_H

#include "source_lcao/module_fde/coupling/fde_diabatic_assembler.h"
#include "source_lcao/module_fde/coupling/fde_linearized_state.h"
#include "source_lcao/module_fde/runtime/fde_runtime_config.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

template <typename Scalar>
struct ScalarOrthogonalizedPairCoupling
{
    std::size_t first_state;
    std::size_t second_state;
    Scalar coupling_ry;
};

template <typename Scalar>
struct ScalarDiabaticPostprocessResult
{
    ScalarFdeDiabaticAssemblyResult<Scalar> assembly;
    ScalarNonorthogonalStateSolution<Scalar> adiabatic_solution;
    std::vector<ScalarOrthogonalizedPairCoupling<Scalar>> orthogonalized_pair_couplings;
    std::vector<double> boundary_twist;
    std::string geometry_fingerprint;
    std::string orbital_fingerprint;
    std::string resource_fingerprint;
};

using OrthogonalizedPairCoupling = ScalarOrthogonalizedPairCoupling<double>;
using DiabaticPostprocessResult = ScalarDiabaticPostprocessResult<double>;
using ComplexDiabaticPostprocessResult = ScalarDiabaticPostprocessResult<std::complex<double>>;

class DiabaticPostprocessor
{
  public:
    static DiabaticPostprocessResult evaluate(
        const FdeRuntimeConfig& config,
        const std::vector<DiabaticDeterminantArtifact>& determinants,
        const std::vector<LinearizedStateArtifact>& linearized_states,
        const std::vector<double>& ao_overlap);

    static ComplexDiabaticPostprocessResult evaluate(
        const FdeRuntimeConfig& config,
        const std::vector<ComplexDiabaticDeterminantArtifact>& determinants,
        const std::vector<ComplexLinearizedStateArtifact>& linearized_states,
        const std::vector<std::complex<double>>& ao_overlap);

    /** Read the sidecar and artifacts, evaluate, and write prefix outputs. */
    static DiabaticPostprocessResult run(const std::string& config_path);

    /** Dispatch the declared real/finite-complex protocol for the native CLI. */
    static std::size_t run_and_count_roots(const std::string& config_path,
                                          std::ostream& running_log);
};

} // namespace fde

#endif // FDE_DIABATIC_POSTPROCESS_H
