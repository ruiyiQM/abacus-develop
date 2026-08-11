#ifndef FDE_COUPLING_PROVIDER_FACTORY_H
#define FDE_COUPLING_PROVIDER_FACTORY_H

#include "../fde_electronic_coupling.h"
#include "../fde_linearized_state.h"

#include <memory>
#include <string>
#include <vector>

namespace fde
{

class FdeCouplingProviderFactory
{
  public:
    static std::unique_ptr<TransitionEnergyProvider> create(
        const std::string& provider,
        const std::vector<LinearizedStateArtifact>& states,
        const std::vector<double>& ao_overlap,
        double singular_value_tolerance);
};

} // namespace fde

#endif // FDE_COUPLING_PROVIDER_FACTORY_H
