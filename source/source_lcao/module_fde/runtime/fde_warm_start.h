#ifndef FDE_RUNTIME_WARM_START_H
#define FDE_RUNTIME_WARM_START_H

#include <cstddef>

namespace fde
{

struct FdeWarmStartPlan
{
    int ionic_step;
    bool reuse_ao_density_matrix;
    bool reuse_orbitals;
    const char* mode;
};

/** Map a compatible resident request to ABACUS' electronic restart semantics. */
class FdeWarmStart
{
  public:
    static FdeWarmStartPlan plan(std::size_t request_index);
};

} // namespace fde

#endif // FDE_RUNTIME_WARM_START_H
