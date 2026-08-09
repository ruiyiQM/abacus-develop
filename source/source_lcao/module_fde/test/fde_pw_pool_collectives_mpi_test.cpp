#include "../fde_grid_partition.h"
#include "../fde_pw_pool_collectives.h"
#include "../fde_spin_density.h"
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

TEST(FdePwPoolCollectivesMpi, NormalizesFixedSpinPopulationsIndependently)
{
    int process_count = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ModulePW::PW_Basis basis("cpu", "double");
    initialize_basis(basis, process_count, rank);

    std::vector<double> alpha(static_cast<std::size_t>(basis.nrxx), static_cast<double>(rank + 1));
    std::vector<double> beta(static_cast<std::size_t>(basis.nrxx), 5.0 * static_cast<double>(rank + 1));
    const double volume = 42.0;
    fde::normalize_spin_density(alpha.data(), beta.data(), alpha.size(), 7, 8, volume, 1.0e-6, basis);

    double populations[2] = {0.0, 0.0};
    for (std::size_t point = 0; point < alpha.size(); ++point)
    {
        populations[0] += alpha[point];
        populations[1] += beta[point];
    }
    fde::PwPoolCollectives::sum_in_place(populations, 2, basis);
    const double volume_element = volume / static_cast<double>(basis.nxyz);
    EXPECT_NEAR(populations[0] * volume_element, 7.0, 1.0e-12);
    EXPECT_NEAR(populations[1] * volume_element, 8.0, 1.0e-12);
    EXPECT_NEAR((populations[0] - populations[1]) * volume_element, -1.0, 1.0e-12);
}

TEST(FdePwPoolCollectivesMpi, PreservesScfShapeButClampsArtifactDensity)
{
    int process_count = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ModulePW::PW_Basis basis("cpu", "double");
    initialize_basis(basis, process_count, rank);

    std::vector<double> scf_alpha(static_cast<std::size_t>(basis.nrxx), 1.0);
    std::vector<double> scf_beta(static_cast<std::size_t>(basis.nrxx), 2.0);
    scf_alpha.front() = -0.25;
    std::vector<double> artifact_alpha = scf_alpha;
    std::vector<double> artifact_beta = scf_beta;

    fde::normalize_spin_density(scf_alpha.data(), scf_beta.data(), scf_alpha.size(), 7, 8, 42.0, 1.0e-6, basis);
    fde::normalize_nonnegative_spin_density(artifact_alpha.data(),
                                            artifact_beta.data(),
                                            artifact_alpha.size(),
                                            7,
                                            8,
                                            42.0,
                                            basis);

    EXPECT_LT(scf_alpha.front(), 0.0);
    EXPECT_DOUBLE_EQ(artifact_alpha.front(), 0.0);
}

TEST(FdePwPoolCollectivesMpi, LeavesSubMicroelectronQuadratureErrorUnchanged)
{
    int process_count = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ModulePW::PW_Basis basis("cpu", "double");
    initialize_basis(basis, process_count, rank);

    const double volume = 42.0;
    std::vector<double> alpha(static_cast<std::size_t>(basis.nrxx), (7.0 + 5.0e-7) / volume);
    std::vector<double> beta(static_cast<std::size_t>(basis.nrxx), (8.0 - 5.0e-7) / volume);
    const std::vector<double> original_alpha = alpha;
    const std::vector<double> original_beta = beta;

    fde::normalize_spin_density(alpha.data(), beta.data(), alpha.size(), 7, 8, volume, 1.0e-6, basis);

    EXPECT_EQ(alpha, original_alpha);
    EXPECT_EQ(beta, original_beta);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
