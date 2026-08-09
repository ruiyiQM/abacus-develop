#include "../fde_grid_partition.h"
#include "../fde_pw_pool_collectives.h"
#include "source_base/matrix3.h"
#include "source_basis/module_pw/pw_basis.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>

namespace
{

void initialize_basis(ModulePW::PW_Basis& basis, const int process_count, const int rank)
{
    basis.initmpi(process_count, rank, MPI_COMM_WORLD);
    const ModuleBase::Matrix3 lattice(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    basis.initgrids(1.0, lattice, 6, 5, 7);
    basis.initparameters(false, 1.0e6, 1, true);
    basis.setuptransform();
}

} // namespace

TEST(FdePwPoolCollectivesMpi, ReducesEnergyComponentsOnEveryRank)
{
    int process_count = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ModulePW::PW_Basis basis("cpu", "double");
    initialize_basis(basis, process_count, rank);

    double values[3] = {static_cast<double>(rank + 1), 2.0 * static_cast<double>(rank + 1), -static_cast<double>(rank)};
    fde::PwPoolCollectives::sum_in_place(values, 3, basis);
    const double triangular = 0.5 * static_cast<double>(process_count * (process_count + 1));
    EXPECT_DOUBLE_EQ(values[0], triangular);
    EXPECT_DOUBLE_EQ(values[1], 2.0 * triangular);
    EXPECT_DOUBLE_EQ(values[2], -0.5 * static_cast<double>(process_count * (process_count - 1)));
    EXPECT_DOUBLE_EQ(fde::PwPoolCollectives::maximum(static_cast<double>(rank), basis),
                     static_cast<double>(process_count - 1));
}

TEST(FdePwPoolCollectivesMpi, GathersZSlabsInCanonicalArtifactOrder)
{
    int process_count = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ModulePW::PW_Basis basis("cpu", "double");
    initialize_basis(basis, process_count, rank);

    std::vector<double> local(static_cast<std::size_t>(basis.nrxx), 0.0);
    for (int xy = 0; xy < basis.nxy; ++xy)
    {
        for (int local_z = 0; local_z < basis.nplane; ++local_z)
        {
            local[static_cast<std::size_t>(xy * basis.nplane + local_z)]
                = 100.0 * static_cast<double>(xy) + static_cast<double>(basis.startz_current + local_z);
        }
    }
    const std::vector<double> global = fde::DensityGridPartition::gather_to_root(local.data(), basis);
    if (rank != 0)
    {
        EXPECT_TRUE(global.empty());
        return;
    }
    ASSERT_EQ(global.size(), static_cast<std::size_t>(basis.nxyz));
    for (int xy = 0; xy < basis.nxy; ++xy)
    {
        for (int z = 0; z < basis.nz; ++z)
        {
            EXPECT_DOUBLE_EQ(global[static_cast<std::size_t>(xy * basis.nz + z)],
                             100.0 * static_cast<double>(xy) + static_cast<double>(z));
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
