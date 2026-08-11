#ifndef FDE_COUPLING_POLICY_H
#define FDE_COUPLING_POLICY_H

#include <string>

namespace fde
{

/** Canonical names for runtime-selectable diabatic coupling providers. */
class FdeCouplingPolicy
{
  public:
    /** Canonicalize linearized to symmetric_linearized and reject others. */
    static std::string canonical_provider(const std::string& name);
};

} // namespace fde

#endif // FDE_COUPLING_POLICY_H
