#include "source_lcao/module_fde/embedding/fde_grid_partition.h"

#include "source_lcao/module_fde/io/fde_density_artifact.h"
#include "source_basis/module_pw/pw_basis.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace fde
{

namespace
{

std::size_t checked_product(const std::size_t left,
                            const std::size_t right,
                            const char* label)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::invalid_argument(label);
    }
    return left * right;
}

} // namespace

DensityGridPartition::DensityGridPartition(const std::size_t grid_x,
                                           const std::size_t grid_y,
                                           const std::size_t grid_z,
                                           const std::size_t start_z,
                                           const std::size_t local_z)
    : grid_x_(grid_x),
      grid_y_(grid_y),
      grid_z_(grid_z),
      start_z_(start_z),
      local_z_(local_z)
{
    if (grid_x_ == 0 || grid_y_ == 0 || grid_z_ == 0)
    {
        throw std::invalid_argument("FDE density grid dimensions must be positive");
    }
    if (start_z_ > grid_z_ || local_z_ > grid_z_ - start_z_)
    {
        throw std::invalid_argument("FDE density-grid z-slab lies outside the global grid");
    }
    (void)this->global_size();
    (void)this->local_size();
}

DensityGridPartition DensityGridPartition::from_pw_basis(
    const FrozenDensityArtifact& artifact,
    const ModulePW::PW_Basis& basis)
{
    if (basis.nx <= 0 || basis.ny <= 0 || basis.nz <= 0
        || basis.startz_current < 0 || basis.nplane < 0 || basis.nrxx < 0)
    {
        throw std::invalid_argument("FDE received an invalid PW_Basis density layout");
    }
    if (artifact.grid_x != static_cast<std::size_t>(basis.nx)
        || artifact.grid_y != static_cast<std::size_t>(basis.ny)
        || artifact.grid_z != static_cast<std::size_t>(basis.nz))
    {
        throw std::invalid_argument("FDE density artifact does not match the PW_Basis grid");
    }
    DensityGridPartition partition(artifact.grid_x,
                                   artifact.grid_y,
                                   artifact.grid_z,
                                   static_cast<std::size_t>(basis.startz_current),
                                   static_cast<std::size_t>(basis.nplane));
    if (partition.local_size() != static_cast<std::size_t>(basis.nrxx))
    {
        throw std::invalid_argument("FDE PW_Basis nrxx does not match its local z-slab");
    }
    return partition;
}

std::size_t DensityGridPartition::grid_x() const
{
    return grid_x_;
}

std::size_t DensityGridPartition::grid_y() const
{
    return grid_y_;
}

std::size_t DensityGridPartition::grid_z() const
{
    return grid_z_;
}

std::size_t DensityGridPartition::start_z() const
{
    return start_z_;
}

std::size_t DensityGridPartition::local_z() const
{
    return local_z_;
}

std::size_t DensityGridPartition::global_size() const
{
    return checked_product(checked_product(grid_x_,
                                           grid_y_,
                                           "FDE density grid size overflows size_t"),
                           grid_z_,
                           "FDE density grid size overflows size_t");
}

std::size_t DensityGridPartition::local_size() const
{
    return checked_product(checked_product(grid_x_,
                                           grid_y_,
                                           "FDE local density-grid size overflows size_t"),
                           local_z_,
                           "FDE local density-grid size overflows size_t");
}

std::vector<double> DensityGridPartition::extract(
    const std::vector<double>& global_density) const
{
    std::vector<double> local_density(this->local_size());
    this->extract_into(global_density, local_density.data(), local_density.size());
    return local_density;
}

void DensityGridPartition::extract_into(const std::vector<double>& global_density,
                                        double* local_density,
                                        const std::size_t local_size) const
{
    if (global_density.size() != this->global_size())
    {
        throw std::invalid_argument("FDE density array does not match the global artifact grid");
    }
    if (local_size != this->local_size()
        || (local_size != 0 && local_density == nullptr))
    {
        throw std::invalid_argument("FDE density destination does not match the local z-slab");
    }
    if (local_size == 0)
    {
        return;
    }
    for (std::size_t xy = 0; xy < grid_x_ * grid_y_; ++xy)
    {
        const std::size_t global_begin = xy * grid_z_ + start_z_;
        const std::size_t local_begin = xy * local_z_;
        std::copy(global_density.begin() + global_begin,
                  global_density.begin() + global_begin + local_z_,
                  local_density + local_begin);
    }
}

std::vector<double> DensityGridPartition::scatter_from_root(
    const std::vector<double>& global_density,
    const ModulePW::PW_Basis& basis,
    const int root)
{
    if (basis.nx <= 0 || basis.ny <= 0 || basis.nz <= 0
        || basis.nrxx < 0 || basis.nplane < 0
        || static_cast<std::size_t>(basis.nrxx)
               != static_cast<std::size_t>(basis.nx)
                      * static_cast<std::size_t>(basis.ny)
                      * static_cast<std::size_t>(basis.nplane))
    {
        throw std::invalid_argument("FDE cannot scatter an invalid PW_Basis density slab");
    }
    if (root < 0 || root >= basis.poolnproc)
    {
        throw std::invalid_argument("FDE density scatter root is outside the PW pool");
    }

    const std::size_t xy_size
        = static_cast<std::size_t>(basis.nx) * static_cast<std::size_t>(basis.ny);
    const std::size_t global_size = xy_size * static_cast<std::size_t>(basis.nz);
#ifdef __MPI
    if (basis.pool_world == MPI_COMM_NULL || basis.startz == nullptr
        || basis.numz == nullptr)
    {
        throw std::invalid_argument("FDE density scatter requires an initialized PW communicator");
    }
    if ((basis.poolrank == root && global_density.size() != global_size)
        || (basis.poolrank != root && !global_density.empty()))
    {
        throw std::invalid_argument(
            "FDE density scatter requires a global artifact only on root");
    }

    std::vector<int> counts(static_cast<std::size_t>(basis.poolnproc), 0);
    std::vector<int> displacements(static_cast<std::size_t>(basis.poolnproc), 0);
    int packed_size = 0;
    for (int rank = 0; rank < basis.poolnproc; ++rank)
    {
        const std::size_t count
            = xy_size * static_cast<std::size_t>(basis.numz[rank]);
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())
            || packed_size > std::numeric_limits<int>::max() - static_cast<int>(count))
        {
            throw std::overflow_error("FDE density scatter exceeds MPI integer counts");
        }
        counts[rank] = static_cast<int>(count);
        displacements[rank] = packed_size;
        packed_size += counts[rank];
    }
    if (static_cast<std::size_t>(packed_size) != global_size)
    {
        throw std::invalid_argument("FDE PW z-slabs do not cover the global density grid");
    }

    std::vector<double> packed(
        basis.poolrank == root ? global_size : std::size_t{0});
    if (basis.poolrank == root)
    {
        for (int rank = 0; rank < basis.poolnproc; ++rank)
        {
            const std::size_t local_z = static_cast<std::size_t>(basis.numz[rank]);
            const std::size_t start_z = static_cast<std::size_t>(basis.startz[rank]);
            for (std::size_t xy = 0; xy < xy_size; ++xy)
            {
                const std::size_t global_begin
                    = xy * static_cast<std::size_t>(basis.nz) + start_z;
                const std::size_t packed_begin
                    = static_cast<std::size_t>(displacements[rank]) + xy * local_z;
                std::copy(global_density.begin() + global_begin,
                          global_density.begin() + global_begin + local_z,
                          packed.begin() + packed_begin);
            }
        }
    }

    std::vector<double> local(static_cast<std::size_t>(basis.nrxx), 0.0);
    MPI_Scatterv(packed.data(),
                 counts.data(),
                 displacements.data(),
                 MPI_DOUBLE,
                 local.data(),
                 basis.nrxx,
                 MPI_DOUBLE,
                 root,
                 basis.pool_world);
    return local;
#else
    if (basis.poolnproc != 1 || basis.poolrank != 0
        || static_cast<std::size_t>(basis.nrxx) != global_size
        || global_density.size() != global_size)
    {
        throw std::invalid_argument("FDE serial density scatter requires one complete grid");
    }
    return global_density;
#endif
}

std::vector<double> DensityGridPartition::gather_to_root(
    const double* local_density,
    const ModulePW::PW_Basis& basis,
    const int root)
{
    if (basis.nx <= 0 || basis.ny <= 0 || basis.nz <= 0
        || basis.nrxx < 0 || basis.nplane < 0
        || static_cast<std::size_t>(basis.nrxx)
               != static_cast<std::size_t>(basis.nx)
                      * static_cast<std::size_t>(basis.ny)
                      * static_cast<std::size_t>(basis.nplane)
        || (basis.nrxx != 0 && local_density == nullptr))
    {
        throw std::invalid_argument("FDE cannot gather an invalid PW_Basis density slab");
    }
    if (root < 0 || root >= basis.poolnproc)
    {
        throw std::invalid_argument("FDE density gather root is outside the PW pool");
    }

    const std::size_t xy_size
        = static_cast<std::size_t>(basis.nx) * static_cast<std::size_t>(basis.ny);
    const std::size_t global_size = xy_size * static_cast<std::size_t>(basis.nz);
#ifdef __MPI
    if (basis.pool_world == MPI_COMM_NULL || basis.startz == nullptr
        || basis.numz == nullptr)
    {
        throw std::invalid_argument("FDE density gather requires an initialized PW communicator");
    }
    std::vector<int> counts(static_cast<std::size_t>(basis.poolnproc), 0);
    std::vector<int> displacements(static_cast<std::size_t>(basis.poolnproc), 0);
    int gathered_size = 0;
    for (int rank = 0; rank < basis.poolnproc; ++rank)
    {
        const std::size_t count
            = xy_size * static_cast<std::size_t>(basis.numz[rank]);
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())
            || gathered_size > std::numeric_limits<int>::max() - static_cast<int>(count))
        {
            throw std::overflow_error("FDE density gather exceeds MPI integer counts");
        }
        counts[rank] = static_cast<int>(count);
        displacements[rank] = gathered_size;
        gathered_size += counts[rank];
    }
    if (static_cast<std::size_t>(gathered_size) != global_size)
    {
        throw std::invalid_argument("FDE PW z-slabs do not cover the global density grid");
    }

    std::vector<double> gathered(
        basis.poolrank == root ? global_size : std::size_t{0});
    MPI_Gatherv(local_density,
                basis.nrxx,
                MPI_DOUBLE,
                gathered.data(),
                counts.data(),
                displacements.data(),
                MPI_DOUBLE,
                root,
                basis.pool_world);
    if (basis.poolrank != root)
    {
        return std::vector<double>();
    }

    std::vector<double> global(global_size, 0.0);
    for (int rank = 0; rank < basis.poolnproc; ++rank)
    {
        const std::size_t local_z = static_cast<std::size_t>(basis.numz[rank]);
        const std::size_t start_z = static_cast<std::size_t>(basis.startz[rank]);
        for (std::size_t xy = 0; xy < xy_size; ++xy)
        {
            const std::size_t gathered_begin
                = static_cast<std::size_t>(displacements[rank]) + xy * local_z;
            const std::size_t global_begin
                = xy * static_cast<std::size_t>(basis.nz) + start_z;
            std::copy(gathered.begin() + gathered_begin,
                      gathered.begin() + gathered_begin + local_z,
                      global.begin() + global_begin);
        }
    }
    return global;
#else
    if (basis.poolnproc != 1 || basis.poolrank != 0
        || static_cast<std::size_t>(basis.nrxx) != global_size)
    {
        throw std::invalid_argument("FDE serial density gather requires one complete grid");
    }
    return std::vector<double>(local_density, local_density + global_size);
#endif
}

} // namespace fde
