#ifndef FDE_EMBEDDING_PW_POOL_COLLECTIVES_H
#define FDE_EMBEDDING_PW_POOL_COLLECTIVES_H

#include <cstddef>

namespace ModulePW
{
class PW_Basis;
}

namespace fde
{

/** Small, testable collective operations scoped to a PW_Basis pool. */
class PwPoolCollectives
{
  public:
    static void sum_in_place(double* values, std::size_t count, const ModulePW::PW_Basis& basis);
    static double maximum(double value, const ModulePW::PW_Basis& basis);
};

} // namespace fde

#endif // FDE_EMBEDDING_PW_POOL_COLLECTIVES_H
