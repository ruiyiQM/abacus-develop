#ifndef FDE_EMBEDDING_GPU_KERNELS_H
#define FDE_EMBEDDING_GPU_KERNELS_H

#include <cstddef>
#include <vector>

namespace fde
{

enum class KineticFunctional;

struct FdeGpuTransferStats
{
    std::size_t host_to_device_bytes;
    std::size_t device_to_host_bytes;
    std::size_t metadata_uploads;
    std::size_t resident_kinetic_evaluations;
};

/** Per-PotFde CUDA buffers and cuFFT plan retained across electronic steps. */
class FdeGpuWorkspace
{
  public:
    FdeGpuWorkspace();
    ~FdeGpuWorkspace();

    void prepare_spectral(
        std::size_t nx,
        std::size_t ny,
        std::size_t nz,
        const std::vector<int>& box_indices,
        const std::vector<double>& gx,
        const std::vector<double>& gy,
        const std::vector<double>& gz);

    void kinetic_functional(
        const std::vector<double>& density,
        KineticFunctional functional,
        double density_floor,
        double volume_element,
        std::vector<double>& potential,
        double& energy_hartree);

    void spectral_gradient(
        const std::vector<double>& values,
        std::vector<double>& gradient_x,
        std::vector<double>& gradient_y,
        std::vector<double>& gradient_z);

    std::vector<double> spectral_divergence(
        const std::vector<double>& vector_x,
        const std::vector<double>& vector_y,
        const std::vector<double>& vector_z);

    FdeGpuTransferStats transfer_stats() const;

  private:
    class Impl;
    Impl* impl_;

    FdeGpuWorkspace(const FdeGpuWorkspace&);
    FdeGpuWorkspace& operator=(const FdeGpuWorkspace&);
};

void gpu_kinetic_local_terms(
    const std::vector<double>& regularized_density,
    const std::vector<unsigned char>& active_mask,
    const std::vector<double>& gradient_x,
    const std::vector<double>& gradient_y,
    const std::vector<double>& gradient_z,
    KineticFunctional functional,
    double density_floor,
    double volume_element,
    std::vector<double>& potential,
    std::vector<double>& flux_x,
    std::vector<double>& flux_y,
    std::vector<double>& flux_z,
    double& energy_hartree);

void gpu_spectral_gradient(
    const std::vector<double>& values,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    const std::vector<int>& box_indices,
    const std::vector<double>& gx,
    const std::vector<double>& gy,
    const std::vector<double>& gz,
    std::vector<double>& gradient_x,
    std::vector<double>& gradient_y,
    std::vector<double>& gradient_z);

std::vector<double> gpu_spectral_divergence(
    const std::vector<double>& vector_x,
    const std::vector<double>& vector_y,
    const std::vector<double>& vector_z,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    const std::vector<int>& box_indices,
    const std::vector<double>& gx,
    const std::vector<double>& gy,
    const std::vector<double>& gz);

} // namespace fde

#endif // FDE_EMBEDDING_GPU_KERNELS_H
