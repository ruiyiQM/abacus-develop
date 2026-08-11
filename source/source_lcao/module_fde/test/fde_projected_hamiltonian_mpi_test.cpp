#include "../embedding/fde_projected_hamiltonian.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
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

} // namespace memory
} // namespace base_device

namespace
{

class DistributedHamiltonian : public hamilt::Hamilt<double>
{
  public:
    explicit DistributedHamiltonian(const Parallel_2D& distribution)
        : distribution_(distribution),
          hamiltonian_(static_cast<std::size_t>(distribution.get_local_size()), 0.0),
          overlap_(static_cast<std::size_t>(distribution.get_local_size()), 0.0)
    {
        for (int local_column = 0; local_column < distribution_.get_col_size(); ++local_column)
        {
            const int global_column = distribution_.local2global_col(local_column);
            for (int local_row = 0; local_row < distribution_.get_row_size(); ++local_row)
            {
                const int global_row = distribution_.local2global_row(local_row);
                const std::size_t offset
                    = static_cast<std::size_t>(local_row
                                               + local_column * distribution_.get_row_size());
                hamiltonian_[offset]
                    = 100.0 * static_cast<double>(global_column)
                      + static_cast<double>(global_row) + 0.25;
                overlap_[offset]
                    = 10.0 * static_cast<double>(global_column)
                      + static_cast<double>(global_row) + 0.5;
            }
        }
    }

    void matrix(hamilt::MatrixBlock<double>& hamiltonian,
                hamilt::MatrixBlock<double>& overlap) override
    {
        hamiltonian = hamilt::MatrixBlock<double>{
            hamiltonian_.data(),
            static_cast<std::size_t>(distribution_.get_row_size()),
            static_cast<std::size_t>(distribution_.get_col_size()),
            distribution_.get_desc()};
        overlap = hamilt::MatrixBlock<double>{
            overlap_.data(),
            static_cast<std::size_t>(distribution_.get_row_size()),
            static_cast<std::size_t>(distribution_.get_col_size()),
            distribution_.get_desc()};
    }

  private:
    const Parallel_2D& distribution_;
    std::vector<double> hamiltonian_;
    std::vector<double> overlap_;
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
    DistributedHamiltonian full(distribution);
    fde::FdeProjectedHamiltonian projected(full,
                                           distribution,
                                           8,
                                           {0, 2, 5},
                                           1.0e6);

    hamilt::MatrixBlock<double> hamiltonian;
    hamilt::MatrixBlock<double> overlap;
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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
