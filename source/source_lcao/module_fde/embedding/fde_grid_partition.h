#ifndef FDE_EMBEDDING_GRID_PARTITION_H
#define FDE_EMBEDDING_GRID_PARTITION_H

#include <cstddef>
#include <vector>

namespace ModulePW
{
class PW_Basis;
}

namespace fde
{

struct FrozenDensityArtifact;

/**
 * Local z-slab view of an ABACUS real-space density grid.
 *
 * Density artifacts use the same z-contiguous ordering as PW_Basis:
 * global(xy, z) = xy * nz + z and local(xy, z) = xy * nplane + z.
 */
class DensityGridPartition
{
  public:
    DensityGridPartition(std::size_t grid_x,
                         std::size_t grid_y,
                         std::size_t grid_z,
                         std::size_t start_z,
                         std::size_t local_z);

    static DensityGridPartition from_pw_basis(const FrozenDensityArtifact& artifact,
                                               const ModulePW::PW_Basis& basis);

    std::size_t grid_x() const;
    std::size_t grid_y() const;
    std::size_t grid_z() const;
    std::size_t start_z() const;
    std::size_t local_z() const;
    std::size_t global_size() const;
    std::size_t local_size() const;

    std::vector<double> extract(const std::vector<double>& global_density) const;
    void extract_into(const std::vector<double>& global_density,
                      double* local_density,
                      std::size_t local_size) const;

    /** Scatter canonical root density into PW_Basis z-slabs. */
    static std::vector<double> scatter_from_root(
        const std::vector<double>& global_density,
        const ModulePW::PW_Basis& basis,
        int root = 0);

    /** Gather PW_Basis z-slabs and restore canonical xy*nz+z order on root. */
    static std::vector<double> gather_to_root(const double* local_density,
                                              const ModulePW::PW_Basis& basis,
                                              int root = 0);

  private:
    std::size_t grid_x_;
    std::size_t grid_y_;
    std::size_t grid_z_;
    std::size_t start_z_;
    std::size_t local_z_;
};

} // namespace fde

#endif // FDE_EMBEDDING_GRID_PARTITION_H
