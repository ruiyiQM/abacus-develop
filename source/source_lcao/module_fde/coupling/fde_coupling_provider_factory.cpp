#include "fde_coupling_provider_factory.h"

#include "fde_coupling_policy.h"
#include "fde_linearized_state.h"

#include <stdexcept>

namespace fde
{

std::unique_ptr<TransitionEnergyProvider> FdeCouplingProviderFactory::create(
    const std::string& provider,
    const std::vector<LinearizedStateArtifact>& states,
    const std::vector<double>& ao_overlap,
    const double singular_value_tolerance)
{
    const std::string canonical
        = FdeCouplingPolicy::canonical_provider(provider);
    if (canonical == "symmetric_linearized")
    {
        return std::unique_ptr<TransitionEnergyProvider>(
            new LinearizedTransitionEnergy(states,
                                           ao_overlap,
                                           singular_value_tolerance));
    }
    throw std::invalid_argument("FDE coupling provider factory is incomplete");
}

} // namespace fde
