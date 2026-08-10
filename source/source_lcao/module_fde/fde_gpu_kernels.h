#ifndef FDE_GPU_KERNELS_H
#define FDE_GPU_KERNELS_H

#include <cstddef>
#include <vector>

namespace fde
{

enum class KineticFunctional;

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

#endif // FDE_GPU_KERNELS_H
