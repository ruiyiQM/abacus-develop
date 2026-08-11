#ifndef FDE_SESSION_CONTRACT_H
#define FDE_SESSION_CONTRACT_H

namespace fde
{

struct FdeRuntimeConfig;

/** Validate that a new request can reuse one initialized LCAO FDE worker. */
class FdeSessionContract
{
  public:
    static void validate_compatible(const FdeRuntimeConfig& initialized,
                                    const FdeRuntimeConfig& requested);
};

} // namespace fde

#endif // FDE_SESSION_CONTRACT_H
