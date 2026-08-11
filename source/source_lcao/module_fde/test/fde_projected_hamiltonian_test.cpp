#include "../embedding/fde_projected_hamiltonian.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
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

template <>
void synchronize_memory_op<std::complex<double>, DEVICE_CPU, DEVICE_CPU>::operator()(
    std::complex<double>* output,
    const std::complex<double>* input,
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

class ComplexKPointHamiltonian
    : public hamilt::Hamilt<std::complex<double>>
{
  public:
    ComplexKPointHamiltonian()
        : hamiltonians(2, std::vector<std::complex<double>>(16)),
          overlaps(2, std::vector<std::complex<double>>(16))
    {
        for (std::size_t kpoint = 0; kpoint < 2; ++kpoint)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                for (std::size_t row = 0; row < 4; ++row)
                {
                    const std::size_t offset = row + column * 4;
                    hamiltonians[kpoint][offset]
                        = std::complex<double>(10.0 * kpoint + row + column,
                                               row == column ? 0.0 : kpoint + 0.25);
                    overlaps[kpoint][offset]
                        = std::complex<double>(row == column ? 1.0 : 0.01 * (row + column),
                                               row == column ? 0.0 : 0.02 * kpoint);
                }
            }
        }
    }

    void updateHk(const int kpoint) override
    {
        current_kpoint = kpoint;
    }

    void refresh(const bool) override
    {
    }

    void matrix(hamilt::MatrixBlock<std::complex<double>>& h,
                hamilt::MatrixBlock<std::complex<double>>& s) override
    {
        h = hamilt::MatrixBlock<std::complex<double>>{
            hamiltonians.at(static_cast<std::size_t>(current_kpoint)).data(),
            4,
            4,
            nullptr};
        s = hamilt::MatrixBlock<std::complex<double>>{
            overlaps.at(static_cast<std::size_t>(current_kpoint)).data(),
            4,
            4,
            nullptr};
    }

    std::vector<std::vector<std::complex<double>>> hamiltonians;
    std::vector<std::vector<std::complex<double>>> overlaps;
    int current_kpoint = 0;
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

TEST(FdeProjectedHamiltonian, PreservesFactorizedOverlapWithinOneSolve)
{
    DenseHamiltonian full;
    Parallel_2D distribution;
    distribution.set_serial(4, 4);
    fde::FdeProjectedHamiltonian projected(full,
                                           distribution,
                                           4,
                                           {0, 2},
                                           1.0e6);
    hamilt::MatrixBlock<double> h;
    hamilt::MatrixBlock<double> s;
    projected.matrix(h, s);

    // Generalized ELPA factorizes S in place after solving alpha.  The beta
    // call must see that same storage, not a newly reconstructed raw overlap.
    s.p[0 + 0 * 4] = 7.0;
    s.p[2 + 0 * 4] = 0.25;
    projected.matrix(h, s);

    EXPECT_DOUBLE_EQ(s.p[0 + 0 * 4], 7.0);
    EXPECT_DOUBLE_EQ(s.p[2 + 0 * 4], 0.25);
    EXPECT_DOUBLE_EQ(h.p[2 + 0 * 4], 0.2);
}

TEST(FdeProjectedHamiltonian, CanShareGammaOverlapAcrossSpinKPointIndices)
{
    DenseHamiltonian full;
    Parallel_2D distribution;
    distribution.set_serial(4, 4);
    fde::FdeProjectedHamiltonian projected(full,
                                           distribution,
                                           4,
                                           {0, 2},
                                           1.0e6,
                                           2,
                                           {0, 0});
    hamilt::MatrixBlock<double> h;
    hamilt::MatrixBlock<double> s;
    projected.updateHk(0);
    projected.matrix(h, s);
    s.p[0] = 8.0;
    projected.updateHk(1);
    projected.matrix(h, s);

    EXPECT_EQ(full.last_kpoint, 1);
    EXPECT_DOUBLE_EQ(s.p[0], 8.0);
}

TEST(FdeProjectedHamiltonian, KeepsIndependentComplexOverlapForEachKPoint)
{
    ComplexKPointHamiltonian full;
    Parallel_2D distribution;
    distribution.set_serial(4, 4);
    fde::FdeProjectedHamiltonianComplex projected(full,
                                                  distribution,
                                                  4,
                                                  {0, 2},
                                                  1.0e6,
                                                  2);
    hamilt::MatrixBlock<std::complex<double>> h;
    hamilt::MatrixBlock<std::complex<double>> s;

    projected.updateHk(0);
    projected.matrix(h, s);
    s.p[0] = std::complex<double>(7.0, 0.5);

    projected.updateHk(1);
    projected.matrix(h, s);
    EXPECT_EQ(h.p[2], std::complex<double>(12.0, 1.25));
    EXPECT_EQ(s.p[2], std::complex<double>(0.02, 0.02));
    EXPECT_EQ(s.p[1 + 1 * 4], std::complex<double>(1.0, 0.0));
    EXPECT_THROW(projected.updateHk(2), std::out_of_range);

    projected.updateHk(0);
    projected.matrix(h, s);
    EXPECT_EQ(s.p[0], std::complex<double>(7.0, 0.5));
    EXPECT_EQ(h.p[2], std::complex<double>(2.0, 0.25));
}
