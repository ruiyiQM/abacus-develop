#ifndef FDE_COUPLING_PROVIDER_FACTORY_H
#define FDE_COUPLING_PROVIDER_FACTORY_H

#include "fde_electronic_coupling.h"
#include "fde_linearized_state.h"

#include <memory>
#include <string>
#include <vector>

namespace fde
{

template <typename Scalar>
class ScalarFdeCouplingProviderFactory
{
  public:
    static std::unique_ptr<ScalarTransitionEnergyProvider<Scalar>> create(
        const std::string& provider,
        const std::vector<ScalarLinearizedStateArtifact<Scalar>>& states,
        const std::vector<Scalar>& ao_overlap,
        double singular_value_tolerance);
};

using FdeCouplingProviderFactory = ScalarFdeCouplingProviderFactory<double>;
using ComplexFdeCouplingProviderFactory = ScalarFdeCouplingProviderFactory<std::complex<double>>;

} // namespace fde

#endif // FDE_COUPLING_PROVIDER_FACTORY_H
