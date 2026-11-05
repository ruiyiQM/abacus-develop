#include "../embedding/fde_projected_hamiltonian.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <complex>
#include <cstddef>
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

// The shared projection source instantiates both real and complex Hamiltonians.
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

template <typename Scalar>
class DistributedHamiltonian : public hamilt::Hamilt<Scalar>
{
  public:
    DistributedHamiltonian(const Parallel_2D& distribution,
                           const std::vector<std::vector<Scalar>>& full_h,
                           const std::vector<std::vector<Scalar>>& full_s)
        : distribution_(distribution), current_kpoint_(0)
    {
        const std::size_t local_size = distribution.get_local_size();
        hamiltonian_.assign(full_h.size(), std::vector<Scalar>(local_size));
        overlap_.assign(full_s.size(), std::vector<Scalar>(local_size));
        for (std::size_t kpoint = 0; kpoint < full_h.size(); ++kpoint)
        {
            for (int local_column = 0; local_column < distribution_.get_col_size(); ++local_column)
            {
                const int global_column = distribution_.local2global_col(local_column);
                for (int local_row = 0; local_row < distribution_.get_row_size(); ++local_row)
                {
                    const int global_row = distribution_.local2global_row(local_row);
                    const std::size_t offset = local_row + local_column * distribution_.get_row_size();
                    const std::size_t global_offset = global_row + global_column * 8;
                    hamiltonian_[kpoint][offset] = full_h[kpoint][global_offset];
                    overlap_[kpoint][offset] = full_s[kpoint][global_offset];
                }
            }
        }
    }

    void updateHk(const int kpoint) override
    {
        current_kpoint_ = kpoint;
    }

    void matrix(ModuleBase::MatrixBlock<Scalar>& hamiltonian,
                ModuleBase::MatrixBlock<Scalar>& overlap) override
    {
        hamiltonian = ModuleBase::MatrixBlock<Scalar>{
            hamiltonian_[current_kpoint_].data(),
            static_cast<std::size_t>(distribution_.get_row_size()),
            static_cast<std::size_t>(distribution_.get_col_size()),
            distribution_.get_desc()};
        overlap = ModuleBase::MatrixBlock<Scalar>{
            overlap_[current_kpoint_].data(),
            static_cast<std::size_t>(distribution_.get_row_size()),
            static_cast<std::size_t>(distribution_.get_col_size()),
            distribution_.get_desc()};
    }

  private:
    const Parallel_2D& distribution_;
    int current_kpoint_;
    std::vector<std::vector<Scalar>> hamiltonian_;
    std::vector<std::vector<Scalar>> overlap_;
};

bool is_active(const int orbital)
{
    return orbital == 0 || orbital == 2 || orbital == 5;
}

} // namespace

TEST(FdeProjectedHamiltonianMpi, ProjectsEachBlockCyclicLocalMatrixEntry)
{
    Parallel_2D distribution;
    ASSERT_EQ(distribution.init(8, 8, 2, MPI_COMM_WORLD), 0);
    std::vector<std::vector<double>> full_h(1, std::vector<double>(64));
    std::vector<std::vector<double>> full_s(1, std::vector<double>(64));
    for (std::size_t column = 0; column < 8; ++column)
    {
        for (std::size_t row = 0; row < 8; ++row)
        {
            const std::size_t offset = row + column * 8;
            full_h[0][offset] = 100.0 * column + row + 0.25;
            full_s[0][offset] = 10.0 * column + row + 0.5;
        }
    }
    DistributedHamiltonian<double> full(distribution, full_h, full_s);
    fde::FdeProjectedHamiltonian projected(full,
                                           distribution,
                                           8,
                                           {0, 2, 5},
                                           1.0e6);

    ModuleBase::MatrixBlock<double> hamiltonian;
    ModuleBase::MatrixBlock<double> overlap;
    projected.matrix(hamiltonian, overlap);
    EXPECT_EQ(hamiltonian.row,
              static_cast<std::size_t>(distribution.get_row_size()));
    EXPECT_EQ(hamiltonian.col,
              static_cast<std::size_t>(distribution.get_col_size()));
    EXPECT_EQ(hamiltonian.desc, distribution.get_desc());

    for (int local_column = 0; local_column < distribution.get_col_size(); ++local_column)
    {
        const int global_column = distribution.local2global_col(local_column);
        for (int local_row = 0; local_row < distribution.get_row_size(); ++local_row)
        {
            const int global_row = distribution.local2global_row(local_row);
            const std::size_t offset
                = static_cast<std::size_t>(local_row
                                           + local_column * distribution.get_row_size());
            if (is_active(global_row) && is_active(global_column))
            {
                EXPECT_DOUBLE_EQ(hamiltonian.p[offset],
                                 100.0 * static_cast<double>(global_column)
                                     + static_cast<double>(global_row) + 0.25);
                EXPECT_DOUBLE_EQ(overlap.p[offset],
                                 10.0 * static_cast<double>(global_column)
                                     + static_cast<double>(global_row) + 0.5);
            }
            else if (global_row == global_column)
            {
                EXPECT_DOUBLE_EQ(hamiltonian.p[offset], 1.0e6);
                EXPECT_DOUBLE_EQ(overlap.p[offset], 1.0);
            }
            else
            {
                EXPECT_DOUBLE_EQ(hamiltonian.p[offset], 0.0);
                EXPECT_DOUBLE_EQ(overlap.p[offset], 0.0);
            }
        }
    }
}

TEST(FdeProjectedHamiltonianMpi, PreservesComplexHermitianBlocksAndIndependentKPointOverlap)
{
    Parallel_2D distribution;
    ASSERT_EQ(distribution.init(8, 8, 2, MPI_COMM_WORLD), 0);
    typedef std::complex<double> Complex;
    std::vector<std::vector<Complex>> full_h(2, std::vector<Complex>(64));
    std::vector<std::vector<Complex>> full_s(2, std::vector<Complex>(64));
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        for (int column = 0; column < 8; ++column)
        {
            for (int row = 0; row < 8; ++row)
            {
                const std::size_t offset = row + column * 8;
                const double real_h = kpoint + row + column + 0.25;
                const double imaginary_h = 0.125 * (row - column);
                const double real_s = row == column ? 1.0 : 0.001 * (row + column);
                const double imaginary_s = 0.001 * (kpoint + 1) * (row - column);
                full_h[kpoint][offset] = Complex(real_h, imaginary_h);
                full_s[kpoint][offset] = Complex(real_s, imaginary_s);
            }
        }
    }
    DistributedHamiltonian<Complex> full(distribution, full_h, full_s);
    const std::vector<std::size_t> active = {0, 2, 5};
    const std::vector<std::size_t> slots = {0, 1};
    fde::FdeProjectedHamiltonianComplex projected(full, distribution, 8, active, 1.0e6, 2, slots);
    const Complex factorized_marker(7.0, 0.5);
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        projected.updateHk(kpoint);
        ModuleBase::MatrixBlock<Complex> hamiltonian;
        ModuleBase::MatrixBlock<Complex> overlap;
        projected.matrix(hamiltonian, overlap);
        EXPECT_EQ(hamiltonian.desc, distribution.get_desc());
        EXPECT_EQ(overlap.desc, distribution.get_desc());
        for (int column = 0; column < distribution.get_col_size(); ++column)
        {
            const int global_column = distribution.local2global_col(column);
            for (int row = 0; row < distribution.get_row_size(); ++row)
            {
                const int global_row = distribution.local2global_row(row);
                const std::size_t offset = row + column * distribution.get_row_size();
                const std::size_t global_offset = global_row + global_column * 8;
                Complex expected_h(0.0);
                Complex expected_s(0.0);
                if (is_active(global_row) && is_active(global_column))
                {
                    expected_h = full_h[kpoint][global_offset];
                    expected_s = full_s[kpoint][global_offset];
                }
                else if (global_row == global_column)
                {
                    expected_h = Complex(1.0e6);
                    expected_s = Complex(1.0);
                }
                EXPECT_EQ(hamiltonian.p[offset], expected_h);
                EXPECT_EQ(overlap.p[offset], expected_s);
            }
        }
        const std::size_t local_size = overlap.row * overlap.col;
        ASSERT_GT(local_size, 0);
        if (kpoint == 0)
        {
            overlap.p[0] = factorized_marker;
        }
    }
    projected.updateHk(0);
    ModuleBase::MatrixBlock<Complex> hamiltonian;
    ModuleBase::MatrixBlock<Complex> overlap;
    projected.matrix(hamiltonian, overlap);
    EXPECT_EQ(overlap.p[0], factorized_marker);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
