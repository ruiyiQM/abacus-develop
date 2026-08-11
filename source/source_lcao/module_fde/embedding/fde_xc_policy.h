#ifndef FDE_EMBEDDING_XC_POLICY_H
#define FDE_EMBEDDING_XC_POLICY_H

#include <string>

namespace fde
{

/** Functional split between the fragment solver and nonadditive embedding. */
class FdeXcPolicy
{
  public:
    /** Return pbe, pbe0, or scan; reject unsupported fragment functionals. */
    static std::string canonical_fragment(const std::string& name);

    /** Return pbe; other nonadditive XC models are not implemented. */
    static std::string canonical_embedding(const std::string& name);
};

} // namespace fde

#endif // FDE_EMBEDDING_XC_POLICY_H
