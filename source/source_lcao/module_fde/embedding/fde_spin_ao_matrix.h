#ifndef FDE_EMBEDDING_SPIN_AO_MATRIX_H
#define FDE_EMBEDDING_SPIN_AO_MATRIX_H

#include <vector>

namespace fde
{

/** Dense real Gamma-point AO matrices in alpha/beta channel order. */
struct SpinAoMatrix
{
    std::vector<double> alpha;
    std::vector<double> beta;
};

} // namespace fde

#endif // FDE_EMBEDDING_SPIN_AO_MATRIX_H
