#include "../fde_gpu_kernels.h"
#include "../fde_semilocal_functional.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <complex>
#include <mutex>
#include <stdexcept>
#include <string>

namespace fde
{

namespace
{

const int threads_per_block = 256;

void cuda_check(const cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string("FDE CUDA failure in ") + operation
                                 + ": " + cudaGetErrorString(status));
    }
}

void cufft_check(const cufftResult status, const char* operation)
{
    if (status != CUFFT_SUCCESS)
    {
        throw std::runtime_error(std::string("FDE cuFFT failure in ")
                                 + operation + ": code "
                                 + std::to_string(static_cast<int>(status)));
    }
}

int checked_size(const std::size_t size, const char* name)
{
    if (size == 0 || size > static_cast<std::size_t>(INT_MAX))
    {
        throw std::invalid_argument(std::string("FDE GPU ") + name
                                    + " size is invalid");
    }
    return static_cast<int>(size);
}

int blocks_for(const int size)
{
    return std::min((size + threads_per_block - 1) / threads_per_block,
                    65535);
}

template <typename T>
class DeviceBuffer
{
  public:
    DeviceBuffer() : pointer_(nullptr), capacity_(0) {}

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr)
        {
            (void)cudaFree(pointer_);
        }
    }

    void resize(const std::size_t size)
    {
        if (size <= capacity_)
        {
            return;
        }
        if (pointer_ != nullptr)
        {
            cuda_check(cudaFree(pointer_), "buffer resize free");
            pointer_ = nullptr;
            capacity_ = 0;
        }
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                              size * sizeof(T)),
                   "buffer allocation");
        capacity_ = size;
    }

    T* data() { return pointer_; }
    const T* data() const { return pointer_; }
    std::size_t capacity() const { return capacity_; }

  private:
    T* pointer_;
    std::size_t capacity_;

    DeviceBuffer(const DeviceBuffer&);
    DeviceBuffer& operator=(const DeviceBuffer&);
};

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination,
                    const std::vector<T>& source,
                    const char* operation)
{
    destination.resize(source.size());
    cuda_check(cudaMemcpy(destination.data(),
                          source.data(),
                          source.size() * sizeof(T),
                          cudaMemcpyHostToDevice),
               operation);
}

template <typename T>
void copy_to_host(std::vector<T>& destination,
                  const DeviceBuffer<T>& source,
                  const std::size_t size,
                  const char* operation)
{
    destination.resize(size);
    cuda_check(cudaMemcpy(destination.data(),
                          source.data(),
                          size * sizeof(T),
                          cudaMemcpyDeviceToHost),
               operation);
}

__device__ void kinetic_enhancement_device(const int functional,
                                            const double s,
                                            double& enhancement,
                                            double& derivative)
{
    if (functional == static_cast<int>(KineticFunctional::ThomasFermi))
    {
        enhancement = 1.0;
        derivative = 0.0;
        return;
    }
    if (functional == static_cast<int>(KineticFunctional::RevApbek))
    {
        const double kappa = 1.245;
        const double mu = 0.23889;
        const double s2 = s * s;
        const double denominator = kappa + mu * s2;
        enhancement = 1.0 + kappa * mu * s2 / denominator;
        derivative = 2.0 * kappa * kappa * mu * s
                     / (denominator * denominator);
        return;
    }

    const double a = 0.093907;
    const double b = 76.320;
    const double c = 0.26608;
    const double d = -0.0809615;
    const double f = 0.000057767;
    const double alpha = 100.0;
    const double s2 = s * s;
    const double exponential = exp(-alpha * s2);
    const double s4 = s2 * s2;
    const double numerator = (c + d * exponential) * s2 - f * s4;
    const double denominator = 1.0 + a * s * asinh(b * s) + f * s4;
    const double numerator_derivative
        = 2.0 * s * (c + d * exponential)
          - 2.0 * alpha * d * s * s2 * exponential
          - 4.0 * f * s * s2;
    const double denominator_derivative
        = a * (asinh(b * s) + b * s / sqrt(1.0 + b * b * s2))
          + 4.0 * f * s * s2;
    enhancement = 1.0 + numerator / denominator;
    derivative = (numerator_derivative * denominator
                  - numerator * denominator_derivative)
                 / (denominator * denominator);
}

__global__ void kinetic_local_kernel(
    const int size,
    const int functional,
    const double* density,
    const unsigned char* active,
    const double* gradient_x,
    const double* gradient_y,
    const double* gradient_z,
    const double density_floor,
    const double volume_element,
    double* potential,
    double* flux_x,
    double* flux_y,
    double* flux_z,
    double* energy)
{
    const double pi = 3.141592653589793238462643383279502884;
    const double c_tf = 0.3 * pow(3.0 * pi * pi, 2.0 / 3.0);
    const double reduced_gradient_scale
        = 1.0 / (2.0 * pow(3.0 * pi * pi, 1.0 / 3.0));
    const double reference_energy_density
        = c_tf * pow(density_floor, 5.0 / 3.0);
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < size;
         index += stride)
    {
        const double rho = density[index];
        double gradient_norm = 0.0;
        if (functional != static_cast<int>(KineticFunctional::ThomasFermi))
        {
            gradient_norm = sqrt(gradient_x[index] * gradient_x[index]
                                 + gradient_y[index] * gradient_y[index]
                                 + gradient_z[index] * gradient_z[index]);
        }
        const double reduced_gradient
            = reduced_gradient_scale * gradient_norm / pow(rho, 4.0 / 3.0);
        double enhancement = 0.0;
        double enhancement_derivative = 0.0;
        kinetic_enhancement_device(functional,
                                   reduced_gradient,
                                   enhancement,
                                   enhancement_derivative);
        energy[index]
            = (c_tf * pow(rho, 5.0 / 3.0) * enhancement
               - reference_energy_density)
              * volume_element;
        potential[index]
            = active[index]
                  ? c_tf * pow(rho, 2.0 / 3.0)
                        * (5.0 * enhancement / 3.0
                           - 4.0 * reduced_gradient
                                 * enhancement_derivative / 3.0)
                  : 0.0;
        if (functional != static_cast<int>(KineticFunctional::ThomasFermi))
        {
            double flux_scale = 0.0;
            if (gradient_norm > 0.0)
            {
                flux_scale
                    = c_tf * reduced_gradient_scale * pow(rho, 1.0 / 3.0)
                      * enhancement_derivative / gradient_norm;
            }
            flux_x[index] = flux_scale * gradient_x[index];
            flux_y[index] = flux_scale * gradient_y[index];
            flux_z[index] = flux_scale * gradient_z[index];
        }
    }
}

__global__ void real_to_complex_kernel(const int size,
                                       const double* input,
                                       cufftDoubleComplex* output)
{
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < size;
         index += stride)
    {
        output[index].x = input[index];
        output[index].y = 0.0;
    }
}

__global__ void regularize_density_kernel(const int size,
                                          const double* input,
                                          const double density_floor,
                                          double* regularized,
                                          unsigned char* active)
{
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < size;
         index += stride)
    {
        active[index] = input[index] > density_floor ? 1 : 0;
        regularized[index] = fmax(input[index], density_floor);
    }
}

__global__ void subtract_divergence_kernel(const int size,
                                           const unsigned char* active,
                                           const double* divergence,
                                           double* potential)
{
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < size;
         index += stride)
    {
        potential[index] = active[index]
                               ? potential[index] - divergence[index]
                               : 0.0;
    }
}

__global__ void spectral_derivative_kernel(
    const int plane_waves,
    const int* box_indices,
    const double* g_component,
    const cufftDoubleComplex* transformed,
    cufftDoubleComplex* derivative,
    const bool add)
{
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < plane_waves;
         index += stride)
    {
        const int box = box_indices[index];
        const double g = g_component[index];
        const cufftDoubleComplex value = transformed[box];
        const cufftDoubleComplex term = make_cuDoubleComplex(-g * value.y,
                                                              g * value.x);
        if (add)
        {
            derivative[box].x += term.x;
            derivative[box].y += term.y;
        }
        else
        {
            derivative[box] = term;
        }
    }
}

__global__ void complex_to_real_kernel(const int size,
                                       const double inverse_size,
                                       const cufftDoubleComplex* input,
                                       double* output)
{
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < size;
         index += stride)
    {
        output[index] = inverse_size * input[index].x;
    }
}

class PointWorkspace
{
  public:
    DeviceBuffer<double> raw_density;
    DeviceBuffer<double> density;
    DeviceBuffer<unsigned char> active;
    DeviceBuffer<double> gradient_x;
    DeviceBuffer<double> gradient_y;
    DeviceBuffer<double> gradient_z;
    DeviceBuffer<double> potential;
    DeviceBuffer<double> flux_x;
    DeviceBuffer<double> flux_y;
    DeviceBuffer<double> flux_z;
    DeviceBuffer<double> divergence;
    DeviceBuffer<double> energy;
};

class SpectralWorkspace
{
  public:
    SpectralWorkspace() : plan_(0), nx_(0), ny_(0), nz_(0) {}

    ~SpectralWorkspace()
    {
        if (plan_ != 0)
        {
            (void)cufftDestroy(plan_);
        }
    }

    void prepare(const int nx, const int ny, const int nz,
                 const std::size_t plane_waves)
    {
        const std::size_t size = static_cast<std::size_t>(nx) * ny * nz;
        real.resize(size);
        transformed.resize(size);
        work.resize(size);
        box_indices.resize(plane_waves);
        gx.resize(plane_waves);
        gy.resize(plane_waves);
        gz.resize(plane_waves);
        if (plan_ == 0 || nx != nx_ || ny != ny_ || nz != nz_)
        {
            if (plan_ != 0)
            {
                cufft_check(cufftDestroy(plan_), "plan replacement");
                plan_ = 0;
            }
            cufft_check(cufftPlan3d(&plan_, nx, ny, nz, CUFFT_Z2Z),
                        "3D plan creation");
            nx_ = nx;
            ny_ = ny;
            nz_ = nz;
        }
    }

    cufftHandle plan() const { return plan_; }

    DeviceBuffer<double> real;
    DeviceBuffer<cufftDoubleComplex> transformed;
    DeviceBuffer<cufftDoubleComplex> work;
    DeviceBuffer<int> box_indices;
    DeviceBuffer<double> gx;
    DeviceBuffer<double> gy;
    DeviceBuffer<double> gz;

  private:
    cufftHandle plan_;
    int nx_;
    int ny_;
    int nz_;
};

std::mutex& gpu_mutex()
{
    static std::mutex mutex;
    return mutex;
}

PointWorkspace& point_workspace()
{
    static PointWorkspace workspace;
    return workspace;
}

SpectralWorkspace& spectral_workspace()
{
    static SpectralWorkspace workspace;
    return workspace;
}

void validate_spectral_arguments(const std::size_t size,
                                 const std::vector<int>& box_indices,
                                 const std::vector<double>& gx,
                                 const std::vector<double>& gy,
                                 const std::vector<double>& gz)
{
    if (box_indices.size() != gx.size() || gx.size() != gy.size()
        || gy.size() != gz.size())
    {
        throw std::invalid_argument("FDE GPU reciprocal-grid arrays differ in size");
    }
    for (std::size_t index = 0; index < box_indices.size(); ++index)
    {
        if (box_indices[index] < 0
            || static_cast<std::size_t>(box_indices[index]) >= size)
        {
            throw std::invalid_argument("FDE GPU reciprocal-grid index is out of range");
        }
    }
}

void upload_spectral_metadata(SpectralWorkspace& workspace,
                              const std::vector<int>& box_indices,
                              const std::vector<double>& gx,
                              const std::vector<double>& gy,
                              const std::vector<double>& gz)
{
    copy_to_device(workspace.box_indices, box_indices, "box-index upload");
    copy_to_device(workspace.gx, gx, "Gx upload");
    copy_to_device(workspace.gy, gy, "Gy upload");
    copy_to_device(workspace.gz, gz, "Gz upload");
}

void forward_real(SpectralWorkspace& workspace,
                  const std::vector<double>& values,
                  const int size)
{
    copy_to_device(workspace.real, values, "real-grid upload");
    real_to_complex_kernel<<<blocks_for(size), threads_per_block>>>(
        size, workspace.real.data(), workspace.transformed.data());
    cuda_check(cudaGetLastError(), "real-to-complex kernel");
    cufft_check(cufftExecZ2Z(workspace.plan(),
                            workspace.transformed.data(),
                            workspace.transformed.data(),
                            CUFFT_FORWARD),
                "forward transform");
}

void forward_device_real(SpectralWorkspace& workspace,
                         const double* values,
                         const int size)
{
    real_to_complex_kernel<<<blocks_for(size), threads_per_block>>>(
        size, values, workspace.transformed.data());
    cuda_check(cudaGetLastError(), "real-to-complex kernel");
    cufft_check(cufftExecZ2Z(workspace.plan(),
                            workspace.transformed.data(),
                            workspace.transformed.data(),
                            CUFFT_FORWARD),
                "forward transform");
}

void inverse_to_device(SpectralWorkspace& workspace,
                       DeviceBuffer<double>& result,
                       const int size)
{
    result.resize(static_cast<std::size_t>(size));
    cufft_check(cufftExecZ2Z(workspace.plan(),
                            workspace.work.data(),
                            workspace.work.data(),
                            CUFFT_INVERSE),
                "inverse transform");
    complex_to_real_kernel<<<blocks_for(size), threads_per_block>>>(
        size, 1.0 / static_cast<double>(size), workspace.work.data(),
        result.data());
    cuda_check(cudaGetLastError(), "complex-to-real kernel");
}

void inverse_to_host(SpectralWorkspace& workspace,
                     std::vector<double>& result,
                     const int size)
{
    inverse_to_device(workspace, workspace.real, size);
    copy_to_host(result, workspace.real, static_cast<std::size_t>(size),
                 "real-grid download");
}

} // namespace

class FdeGpuWorkspace::Impl
{
  public:
    Impl()
        : spectral_ready(false),
          nx(0),
          ny(0),
          nz(0),
          stats{0, 0, 0, 0}
    {
    }

    PointWorkspace point;
    SpectralWorkspace spectral;
    bool spectral_ready;
    std::size_t nx;
    std::size_t ny;
    std::size_t nz;
    FdeGpuTransferStats stats;
};

FdeGpuWorkspace::FdeGpuWorkspace() : impl_(new Impl()) {}

FdeGpuWorkspace::~FdeGpuWorkspace()
{
    delete impl_;
}

void FdeGpuWorkspace::prepare_spectral(
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t nz,
    const std::vector<int>& box_indices,
    const std::vector<double>& gx,
    const std::vector<double>& gy,
    const std::vector<double>& gz)
{
    if (impl_->spectral_ready)
    {
        throw std::logic_error(
            "FDE resident GPU reciprocal metadata is immutable");
    }
    const std::size_t size = nx * ny * nz;
    validate_spectral_arguments(size, box_indices, gx, gy, gz);
    checked_size(size, "FFT");
    checked_size(box_indices.size(), "plane-wave");
    impl_->spectral.prepare(static_cast<int>(nx),
                            static_cast<int>(ny),
                            static_cast<int>(nz),
                            box_indices.size());
    upload_spectral_metadata(impl_->spectral,
                             box_indices,
                             gx,
                             gy,
                             gz);
    impl_->stats.host_to_device_bytes
        += box_indices.size() * sizeof(int)
           + (gx.size() + gy.size() + gz.size()) * sizeof(double);
    ++impl_->stats.metadata_uploads;
    impl_->nx = nx;
    impl_->ny = ny;
    impl_->nz = nz;
    impl_->spectral_ready = true;
}

void FdeGpuWorkspace::kinetic_functional(
    const std::vector<double>& density,
    const KineticFunctional functional,
    const double density_floor,
    const double volume_element,
    std::vector<double>& potential,
    double& energy_hartree)
{
    if (functional != KineticFunctional::ThomasFermi
        && functional != KineticFunctional::Pw91k
        && functional != KineticFunctional::RevApbek)
    {
        throw std::invalid_argument(
            "FDE resident GPU received an unknown kinetic functional");
    }
    const bool local = functional == KineticFunctional::ThomasFermi;
    if (!local
        && (!impl_->spectral_ready
            || density.size() != impl_->nx * impl_->ny * impl_->nz))
    {
        throw std::invalid_argument(
            "FDE resident GPU GGA density does not fill its FFT box");
    }
    const std::size_t size = density.size();
    const int count = checked_size(size, "resident NAKE");
    PointWorkspace& point = impl_->point;
    copy_to_device(point.raw_density, density, "resident density upload");
    impl_->stats.host_to_device_bytes += size * sizeof(double);
    point.density.resize(size);
    point.active.resize(size);
    point.potential.resize(size);
    point.energy.resize(size);
    regularize_density_kernel<<<blocks_for(count), threads_per_block>>>(
        count,
        point.raw_density.data(),
        density_floor,
        point.density.data(),
        point.active.data());
    cuda_check(cudaGetLastError(), "resident density regularization kernel");

    const double* device_gx = nullptr;
    const double* device_gy = nullptr;
    const double* device_gz = nullptr;
    double* device_flux_x = nullptr;
    double* device_flux_y = nullptr;
    double* device_flux_z = nullptr;
    if (!local)
    {
        SpectralWorkspace& spectral = impl_->spectral;
        point.gradient_x.resize(size);
        point.gradient_y.resize(size);
        point.gradient_z.resize(size);
        DeviceBuffer<double>* gradients[3]
            = {&point.gradient_x, &point.gradient_y, &point.gradient_z};
        const DeviceBuffer<double>* g_components[3]
            = {&spectral.gx, &spectral.gy, &spectral.gz};
        const int reciprocal_count
            = checked_size(spectral.gx.capacity(), "plane-wave");
        forward_device_real(spectral, point.density.data(), count);
        for (int direction = 0; direction < 3; ++direction)
        {
            cuda_check(cudaMemset(spectral.work.data(),
                                  0,
                                  size * sizeof(cufftDoubleComplex)),
                       "resident gradient buffer clear");
            spectral_derivative_kernel<<<blocks_for(reciprocal_count),
                                         threads_per_block>>>(
                reciprocal_count,
                spectral.box_indices.data(),
                g_components[direction]->data(),
                spectral.transformed.data(),
                spectral.work.data(),
                false);
            cuda_check(cudaGetLastError(),
                       "resident spectral-gradient kernel");
            inverse_to_device(spectral, *gradients[direction], count);
        }
        point.flux_x.resize(size);
        point.flux_y.resize(size);
        point.flux_z.resize(size);
        device_gx = point.gradient_x.data();
        device_gy = point.gradient_y.data();
        device_gz = point.gradient_z.data();
        device_flux_x = point.flux_x.data();
        device_flux_y = point.flux_y.data();
        device_flux_z = point.flux_z.data();
    }

    kinetic_local_kernel<<<blocks_for(count), threads_per_block>>>(
        count,
        static_cast<int>(functional),
        point.density.data(),
        point.active.data(),
        device_gx,
        device_gy,
        device_gz,
        density_floor,
        volume_element,
        point.potential.data(),
        device_flux_x,
        device_flux_y,
        device_flux_z,
        point.energy.data());
    cuda_check(cudaGetLastError(), "resident NAKE point kernel");

    if (!local)
    {
        SpectralWorkspace& spectral = impl_->spectral;
        const int reciprocal_count
            = checked_size(spectral.gx.capacity(), "plane-wave");
        cuda_check(cudaMemset(spectral.work.data(),
                              0,
                              size * sizeof(cufftDoubleComplex)),
                   "resident divergence buffer clear");
        const DeviceBuffer<double>* fluxes[3]
            = {&point.flux_x, &point.flux_y, &point.flux_z};
        const DeviceBuffer<double>* g_components[3]
            = {&spectral.gx, &spectral.gy, &spectral.gz};
        for (int direction = 0; direction < 3; ++direction)
        {
            forward_device_real(spectral, fluxes[direction]->data(), count);
            spectral_derivative_kernel<<<blocks_for(reciprocal_count),
                                         threads_per_block>>>(
                reciprocal_count,
                spectral.box_indices.data(),
                g_components[direction]->data(),
                spectral.transformed.data(),
                spectral.work.data(),
                true);
            cuda_check(cudaGetLastError(),
                       "resident spectral-divergence kernel");
        }
        inverse_to_device(spectral, point.divergence, count);
        subtract_divergence_kernel<<<blocks_for(count), threads_per_block>>>(
            count,
            point.active.data(),
            point.divergence.data(),
            point.potential.data());
        cuda_check(cudaGetLastError(),
                   "resident divergence assembly kernel");
    }

    thrust::device_ptr<double> energy_begin(point.energy.data());
    energy_hartree = thrust::reduce(energy_begin,
                                    energy_begin + count,
                                    0.0,
                                    thrust::plus<double>());
    copy_to_host(potential,
                 point.potential,
                 size,
                 "resident NAKE potential download");
    impl_->stats.device_to_host_bytes += size * sizeof(double);
    ++impl_->stats.resident_kinetic_evaluations;
}

void FdeGpuWorkspace::spectral_gradient(
    const std::vector<double>& values,
    std::vector<double>& gradient_x,
    std::vector<double>& gradient_y,
    std::vector<double>& gradient_z)
{
    if (!impl_->spectral_ready
        || values.size() != impl_->nx * impl_->ny * impl_->nz)
    {
        throw std::invalid_argument(
            "FDE resident GPU gradient input does not fill its FFT box");
    }
    const int count = checked_size(values.size(), "FFT");
    const int plane_waves
        = checked_size(impl_->spectral.gx.capacity(), "plane-wave");
    forward_real(impl_->spectral, values, count);
    impl_->stats.host_to_device_bytes += values.size() * sizeof(double);
    const DeviceBuffer<double>* components[3]
        = {&impl_->spectral.gx, &impl_->spectral.gy, &impl_->spectral.gz};
    std::vector<double>* outputs[3]
        = {&gradient_x, &gradient_y, &gradient_z};
    for (int direction = 0; direction < 3; ++direction)
    {
        cuda_check(cudaMemset(impl_->spectral.work.data(),
                              0,
                              values.size() * sizeof(cufftDoubleComplex)),
                   "gradient buffer clear");
        spectral_derivative_kernel<<<blocks_for(plane_waves),
                                     threads_per_block>>>(
            plane_waves,
            impl_->spectral.box_indices.data(),
            components[direction]->data(),
            impl_->spectral.transformed.data(),
            impl_->spectral.work.data(),
            false);
        cuda_check(cudaGetLastError(), "spectral-gradient kernel");
        inverse_to_host(impl_->spectral, *outputs[direction], count);
    }
    impl_->stats.device_to_host_bytes
        += 3 * values.size() * sizeof(double);
}

std::vector<double> FdeGpuWorkspace::spectral_divergence(
    const std::vector<double>& vector_x,
    const std::vector<double>& vector_y,
    const std::vector<double>& vector_z)
{
    const std::size_t size = impl_->nx * impl_->ny * impl_->nz;
    if (!impl_->spectral_ready || vector_x.size() != size
        || vector_y.size() != size || vector_z.size() != size)
    {
        throw std::invalid_argument(
            "FDE resident GPU divergence input does not fill its FFT box");
    }
    const int count = checked_size(size, "FFT");
    const int plane_waves
        = checked_size(impl_->spectral.gx.capacity(), "plane-wave");
    cuda_check(cudaMemset(impl_->spectral.work.data(),
                          0,
                          size * sizeof(cufftDoubleComplex)),
               "divergence buffer clear");
    const std::vector<double>* inputs[3]
        = {&vector_x, &vector_y, &vector_z};
    const DeviceBuffer<double>* components[3]
        = {&impl_->spectral.gx, &impl_->spectral.gy, &impl_->spectral.gz};
    for (int direction = 0; direction < 3; ++direction)
    {
        forward_real(impl_->spectral, *inputs[direction], count);
        spectral_derivative_kernel<<<blocks_for(plane_waves),
                                     threads_per_block>>>(
            plane_waves,
            impl_->spectral.box_indices.data(),
            components[direction]->data(),
            impl_->spectral.transformed.data(),
            impl_->spectral.work.data(),
            true);
        cuda_check(cudaGetLastError(), "spectral-divergence kernel");
    }
    std::vector<double> result;
    inverse_to_host(impl_->spectral, result, count);
    impl_->stats.host_to_device_bytes += 3 * size * sizeof(double);
    impl_->stats.device_to_host_bytes += size * sizeof(double);
    return result;
}

FdeGpuTransferStats FdeGpuWorkspace::transfer_stats() const
{
    return impl_->stats;
}

void gpu_kinetic_local_terms(
    const std::vector<double>& regularized_density,
    const std::vector<unsigned char>& active_mask,
    const std::vector<double>& gradient_x,
    const std::vector<double>& gradient_y,
    const std::vector<double>& gradient_z,
    const KineticFunctional functional,
    const double density_floor,
    const double volume_element,
    std::vector<double>& potential,
    std::vector<double>& flux_x,
    std::vector<double>& flux_y,
    std::vector<double>& flux_z,
    double& energy_hartree)
{
    const std::size_t size = regularized_density.size();
    const bool local = functional == KineticFunctional::ThomasFermi;
    if (active_mask.size() != size
        || (!local && (gradient_x.size() != size || gradient_y.size() != size
                       || gradient_z.size() != size)))
    {
        throw std::invalid_argument("FDE GPU NAKE arrays differ in size");
    }
    if (functional != KineticFunctional::ThomasFermi
        && functional != KineticFunctional::Pw91k
        && functional != KineticFunctional::RevApbek)
    {
        throw std::invalid_argument("FDE GPU received an unknown kinetic functional");
    }
    const int count = checked_size(size, "NAKE");
    std::lock_guard<std::mutex> lock(gpu_mutex());
    PointWorkspace& workspace = point_workspace();
    copy_to_device(workspace.density, regularized_density, "density upload");
    copy_to_device(workspace.active, active_mask, "active-mask upload");
    workspace.potential.resize(size);
    workspace.energy.resize(size);
    const double* device_gx = nullptr;
    const double* device_gy = nullptr;
    const double* device_gz = nullptr;
    double* device_flux_x = nullptr;
    double* device_flux_y = nullptr;
    double* device_flux_z = nullptr;
    if (!local)
    {
        copy_to_device(workspace.gradient_x, gradient_x, "gradient-x upload");
        copy_to_device(workspace.gradient_y, gradient_y, "gradient-y upload");
        copy_to_device(workspace.gradient_z, gradient_z, "gradient-z upload");
        workspace.flux_x.resize(size);
        workspace.flux_y.resize(size);
        workspace.flux_z.resize(size);
        device_gx = workspace.gradient_x.data();
        device_gy = workspace.gradient_y.data();
        device_gz = workspace.gradient_z.data();
        device_flux_x = workspace.flux_x.data();
        device_flux_y = workspace.flux_y.data();
        device_flux_z = workspace.flux_z.data();
    }
    kinetic_local_kernel<<<blocks_for(count), threads_per_block>>>(
        count,
        static_cast<int>(functional),
        workspace.density.data(),
        workspace.active.data(),
        device_gx,
        device_gy,
        device_gz,
        density_floor,
        volume_element,
        workspace.potential.data(),
        device_flux_x,
        device_flux_y,
        device_flux_z,
        workspace.energy.data());
    cuda_check(cudaGetLastError(), "NAKE point kernel");
    thrust::device_ptr<double> energy_begin(workspace.energy.data());
    energy_hartree = thrust::reduce(energy_begin,
                                    energy_begin + count,
                                    0.0,
                                    thrust::plus<double>());
    copy_to_host(potential, workspace.potential, size, "NAKE potential download");
    if (local)
    {
        flux_x.clear();
        flux_y.clear();
        flux_z.clear();
    }
    else
    {
        copy_to_host(flux_x, workspace.flux_x, size, "NAKE flux-x download");
        copy_to_host(flux_y, workspace.flux_y, size, "NAKE flux-y download");
        copy_to_host(flux_z, workspace.flux_z, size, "NAKE flux-z download");
    }
}

void gpu_spectral_gradient(
    const std::vector<double>& values,
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t nz,
    const std::vector<int>& box_indices,
    const std::vector<double>& gx,
    const std::vector<double>& gy,
    const std::vector<double>& gz,
    std::vector<double>& gradient_x,
    std::vector<double>& gradient_y,
    std::vector<double>& gradient_z)
{
    const std::size_t size = nx * ny * nz;
    if (values.size() != size)
    {
        throw std::invalid_argument("FDE GPU gradient input does not fill the FFT box");
    }
    validate_spectral_arguments(size, box_indices, gx, gy, gz);
    const int count = checked_size(size, "FFT");
    const int plane_waves = checked_size(box_indices.size(), "plane-wave");
    std::lock_guard<std::mutex> lock(gpu_mutex());
    SpectralWorkspace& workspace = spectral_workspace();
    workspace.prepare(static_cast<int>(nx), static_cast<int>(ny),
                      static_cast<int>(nz), box_indices.size());
    upload_spectral_metadata(workspace, box_indices, gx, gy, gz);
    forward_real(workspace, values, count);
    const DeviceBuffer<double>* components[3]
        = {&workspace.gx, &workspace.gy, &workspace.gz};
    std::vector<double>* outputs[3]
        = {&gradient_x, &gradient_y, &gradient_z};
    for (int direction = 0; direction < 3; ++direction)
    {
        cuda_check(cudaMemset(workspace.work.data(),
                              0,
                              size * sizeof(cufftDoubleComplex)),
                   "gradient buffer clear");
        spectral_derivative_kernel<<<blocks_for(plane_waves), threads_per_block>>>(
            plane_waves,
            workspace.box_indices.data(),
            components[direction]->data(),
            workspace.transformed.data(),
            workspace.work.data(),
            false);
        cuda_check(cudaGetLastError(), "spectral-gradient kernel");
        inverse_to_host(workspace, *outputs[direction], count);
    }
}

std::vector<double> gpu_spectral_divergence(
    const std::vector<double>& vector_x,
    const std::vector<double>& vector_y,
    const std::vector<double>& vector_z,
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t nz,
    const std::vector<int>& box_indices,
    const std::vector<double>& gx,
    const std::vector<double>& gy,
    const std::vector<double>& gz)
{
    const std::size_t size = nx * ny * nz;
    if (vector_x.size() != size || vector_y.size() != size
        || vector_z.size() != size)
    {
        throw std::invalid_argument("FDE GPU divergence input does not fill the FFT box");
    }
    validate_spectral_arguments(size, box_indices, gx, gy, gz);
    const int count = checked_size(size, "FFT");
    const int plane_waves = checked_size(box_indices.size(), "plane-wave");
    std::lock_guard<std::mutex> lock(gpu_mutex());
    SpectralWorkspace& workspace = spectral_workspace();
    workspace.prepare(static_cast<int>(nx), static_cast<int>(ny),
                      static_cast<int>(nz), box_indices.size());
    upload_spectral_metadata(workspace, box_indices, gx, gy, gz);
    cuda_check(cudaMemset(workspace.work.data(),
                          0,
                          size * sizeof(cufftDoubleComplex)),
               "divergence buffer clear");
    const std::vector<double>* inputs[3] = {&vector_x, &vector_y, &vector_z};
    const DeviceBuffer<double>* components[3]
        = {&workspace.gx, &workspace.gy, &workspace.gz};
    for (int direction = 0; direction < 3; ++direction)
    {
        forward_real(workspace, *inputs[direction], count);
        spectral_derivative_kernel<<<blocks_for(plane_waves), threads_per_block>>>(
            plane_waves,
            workspace.box_indices.data(),
            components[direction]->data(),
            workspace.transformed.data(),
            workspace.work.data(),
            true);
        cuda_check(cudaGetLastError(), "spectral-divergence kernel");
    }
    std::vector<double> result;
    inverse_to_host(workspace, result, count);
    return result;
}

} // namespace fde
