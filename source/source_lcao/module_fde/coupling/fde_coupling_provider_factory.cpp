#include "fde_coupling_provider_factory.h"

#include "fde_coupling_policy.h"
#include "fde_linearized_state.h"

#include <stdexcept>

namespace fde
{

template <typename Scalar>
std::unique_ptr<ScalarTransitionEnergyProvider<Scalar>> ScalarFdeCouplingProviderFactory<Scalar>::create(
    const std::string& provider,
    const std::vector<ScalarLinearizedStateArtifact<Scalar>>& states,
    const std::vector<Scalar>& ao_overlap,
    const double singular_value_tolerance)
{
    const std::string canonical
        = FdeCouplingPolicy::canonical_provider(provider);
    if (canonical == "symmetric_linearized")
    {
        return std::unique_ptr<ScalarTransitionEnergyProvider<Scalar>>(
            new ScalarLinearizedTransitionEnergy<Scalar>(states,
                                           ao_overlap,
                                           singular_value_tolerance));
    }
    throw std::invalid_argument("FDE coupling provider factory is incomplete");
}

template class ScalarFdeCouplingProviderFactory<double>;
template class ScalarFdeCouplingProviderFactory<std::complex<double>>;

} // namespace fde
