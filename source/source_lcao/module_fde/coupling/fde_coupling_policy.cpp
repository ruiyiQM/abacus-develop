#include "fde_coupling_policy.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace fde
{

std::string FdeCouplingPolicy::canonical_provider(const std::string& name)
{
    std::string canonical(name);
    std::transform(canonical.begin(),
                   canonical.end(),
                   canonical.begin(),
                   [](const unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    if (canonical == "linearized")
    {
        canonical = "symmetric_linearized";
    }
    if (canonical != "symmetric_linearized")
    {
        throw std::invalid_argument(
            "FDE coupling provider must be symmetric_linearized");
    }
    return canonical;
}

} // namespace fde
