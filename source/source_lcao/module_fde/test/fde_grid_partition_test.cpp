#include "../embedding/fde_grid_partition.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(FdeGridPartition, ExtractsAbacusZContiguousSlab)
{
    std::vector<double> global(16, 0.0);
    for (std::size_t xy = 0; xy < 4; ++xy)
    {
        for (std::size_t z = 0; z < 4; ++z)
        {
            global[xy * 4 + z] = static_cast<double>(100 * xy + z);
        }
    }

    const fde::DensityGridPartition partition(2, 2, 4, 1, 2);
    const std::vector<double> local = partition.extract(global);

    const std::vector<double> expected = {1.0, 2.0,
                                          101.0, 102.0,
                                          201.0, 202.0,
                                          301.0, 302.0};
    EXPECT_EQ(local, expected);
    EXPECT_EQ(partition.global_size(), 16U);
    EXPECT_EQ(partition.local_size(), 8U);
}

TEST(FdeGridPartition, AdjacentSlabsCoverEveryGlobalPointOnce)
{
    std::vector<double> global(18, 0.0);
    for (std::size_t index = 0; index < global.size(); ++index)
    {
        global[index] = static_cast<double>(index);
    }

    const fde::DensityGridPartition first(3, 1, 6, 0, 2);
    const fde::DensityGridPartition second(3, 1, 6, 2, 3);
    const fde::DensityGridPartition third(3, 1, 6, 5, 1);
    const std::vector<double> local_first = first.extract(global);
    const std::vector<double> local_second = second.extract(global);
    const std::vector<double> local_third = third.extract(global);

    std::vector<double> restored(global.size(), -1.0);
    const fde::DensityGridPartition* partitions[] = {&first, &second, &third};
    const std::vector<double>* slabs[] = {&local_first, &local_second, &local_third};
    for (std::size_t rank = 0; rank < 3; ++rank)
    {
        for (std::size_t xy = 0; xy < 3; ++xy)
        {
            for (std::size_t local_z = 0; local_z < partitions[rank]->local_z(); ++local_z)
            {
                restored[xy * 6 + partitions[rank]->start_z() + local_z]
                    = (*slabs[rank])[xy * partitions[rank]->local_z() + local_z];
            }
        }
    }
    EXPECT_EQ(restored, global);
}

TEST(FdeGridPartition, RejectsInvalidLayoutsAndArraySizes)
{
    EXPECT_THROW(fde::DensityGridPartition(2, 2, 4, 3, 2), std::invalid_argument);
    EXPECT_THROW(fde::DensityGridPartition(0, 2, 4, 0, 4), std::invalid_argument);

    const fde::DensityGridPartition partition(2, 2, 4, 0, 1);
    EXPECT_THROW(partition.extract(std::vector<double>(15, 0.0)), std::invalid_argument);
    EXPECT_THROW(partition.extract_into(std::vector<double>(16, 0.0), nullptr, 4),
                 std::invalid_argument);
}

TEST(FdeGridPartition, AllowsAnEmptySlabForAnIdleRank)
{
    const fde::DensityGridPartition partition(2, 2, 4, 4, 0);
    const std::vector<double> local = partition.extract(std::vector<double>(16, 1.0));
    EXPECT_TRUE(local.empty());
}
