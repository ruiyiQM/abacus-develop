#include "../fde_projected_hamiltonian.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace base_device
{
namespace memory
{

template <>
void synchronize_memory_op<double, DEVICE_CPU, DEVICE_CPU>::operator()(
    double* output,
    const double* input,
    const std::size_t size)
{
    std::copy(input, input + size, output);
}

} // namespace memory
} // namespace base_device

namespace
{

class DenseHamiltonian : public hamilt::Hamilt<double>
{
  public:
    DenseHamiltonian()
        : hamiltonian{1.0, 0.1, 0.2, 0.3,
                      0.1, 2.0, 0.4, 0.5,
                      0.2, 0.4, 3.0, 0.6,
                      0.3, 0.5, 0.6, 4.0},
          overlap{1.0, 0.01, 0.02, 0.03,
                  0.01, 1.0, 0.04, 0.05,
                  0.02, 0.04, 1.0, 0.06,
                  0.03, 0.05, 0.06, 1.0}
    {
    }

    void updateHk(const int kpoint) override
    {
        last_kpoint = kpoint;
    }

    void refresh(const bool yes) override
    {
        refreshed = yes;
    }

    void matrix(hamilt::MatrixBlock<double>& h, hamilt::MatrixBlock<double>& s) override
    {
        h = hamilt::MatrixBlock<double>{hamiltonian.data(), 4, 4, nullptr};
        s = hamilt::MatrixBlock<double>{overlap.data(), 4, 4, nullptr};
    }

    std::vector<double> hamiltonian;
    std::vector<double> overlap;
    int last_kpoint = -1;
    bool refreshed = false;
};

} // namespace

TEST(FdeProjectedHamiltonian, RetainsOnlyActivePrincipalSubspace)
{
    DenseHamiltonian full;
    Parallel_2D distribution;
    distribution.set_serial(4, 4);
    fde::FdeProjectedHamiltonian projected(full,
                                           distribution,
                                           4,
                                           {0, 2},
                                           1.0e6);
    projected.updateHk(0);
    hamilt::MatrixBlock<double> h;
    hamilt::MatrixBlock<double> s;
    projected.matrix(h, s);

    EXPECT_EQ(full.last_kpoint, 0);
    EXPECT_DOUBLE_EQ(h.p[0 + 0 * 4], 1.0);
    EXPECT_DOUBLE_EQ(h.p[2 + 0 * 4], 0.2);
    EXPECT_DOUBLE_EQ(h.p[0 + 2 * 4], 0.2);
    EXPECT_DOUBLE_EQ(h.p[2 + 2 * 4], 3.0);
    EXPECT_DOUBLE_EQ(s.p[2 + 0 * 4], 0.02);
    EXPECT_DOUBLE_EQ(h.p[1 + 1 * 4], 1.0e6);
    EXPECT_DOUBLE_EQ(s.p[1 + 1 * 4], 1.0);
    EXPECT_DOUBLE_EQ(h.p[3 + 3 * 4], 1.0e6);
    EXPECT_DOUBLE_EQ(s.p[3 + 3 * 4], 1.0);
    EXPECT_DOUBLE_EQ(h.p[1 + 0 * 4], 0.0);
    EXPECT_DOUBLE_EQ(s.p[2 + 3 * 4], 0.0);
}

TEST(FdeProjectedHamiltonian, RejectsUnsortedOrCompleteSelections)
{
    DenseHamiltonian full;
    Parallel_2D distribution;
    distribution.set_serial(4, 4);
    EXPECT_THROW(fde::FdeProjectedHamiltonian(full,
                                              distribution,
                                              4,
                                              {2, 0},
                                              1.0e6),
                 std::invalid_argument);
    EXPECT_THROW(fde::FdeProjectedHamiltonian(full,
                                              distribution,
                                              4,
                                              {0, 1, 2, 3},
                                              1.0e6),
                 std::invalid_argument);
}
