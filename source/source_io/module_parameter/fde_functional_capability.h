#ifndef FDE_FUNCTIONAL_CAPABILITY_H
#define FDE_FUNCTIONAL_CAPABILITY_H

#include <string>

namespace ModuleIO
{

enum class FdeXcCapability
{
    pbe_semilocal,
    meta_gga_requires_tau,
    hybrid_requires_exact_exchange,
    hybrid_meta_gga_requires_tau_and_exact_exchange,
    unsupported
};

/** Classify an XC functional against the native embedded-SCF FDE contract. */
FdeXcCapability classify_fde_xc_functional(const std::string& functional);

/** True for the fragment-level choices exposed by the native workflow. */
bool supports_fde_fragment_xc(const std::string& functional);

/** Explain why a classified functional cannot be used by native FDE. */
std::string fde_xc_capability_error(FdeXcCapability capability);

} // namespace ModuleIO

#endif // FDE_FUNCTIONAL_CAPABILITY_H
