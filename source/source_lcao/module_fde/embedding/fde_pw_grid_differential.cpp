#include "source_lcao/module_fde/embedding/fde_pw_grid_differential.h"

#include "source_basis/module_pw/pw_basis.h"

#ifdef __CUDA
#include "source_lcao/module_fde/embedding/fde_gpu_kernels.h"
#endif

#include <complex>
#include <stdexcept>

namespace fde
{

PwGridDifferential::PwGridDifferential(const ModulePW::PW_Basis& basis,
                                       const bool use_gpu)
    : basis_(basis),
      use_gpu_(use_gpu),
      use_gpu_fft_(false)
{
    if (basis_.nrxx < 0 || basis_.npw < 0 || basis_.nmaxgr < basis_.npw
        || (basis_.npw != 0 && basis_.gcar == nullptr))
    {
        throw std::invalid_argument("FDE PW grid derivatives require an initialized PW_Basis");
    }
#ifdef __CUDA
    // ABACUS' GPU PW transform is currently a full-box, single-pool-rank
    // implementation. Distributed FDE derivatives retain the MPI FFT path,
    // while their pointwise NAKE algebra can still run on each rank's GPU.
    use_gpu_fft_ = use_gpu_ && basis_.poolnproc == 1
                   && basis_.nrxx == basis_.nxyz;
    if (use_gpu_fft_)
    {
        gpu_box_indices_.resize(static_cast<std::size_t>(basis_.npw));
        gpu_gx_.resize(gpu_box_indices_.size());
        gpu_gy_.resize(gpu_box_indices_.size());
        gpu_gz_.resize(gpu_box_indices_.size());
        for (int reciprocal_index = 0; reciprocal_index < basis_.npw;
             ++reciprocal_index)
        {
            const int isz = basis_.ig2isz[reciprocal_index];
            const int iz = isz % basis_.nz;
            const int stick = isz / basis_.nz;
            const int ixy = basis_.is2fftixy[stick];
            const int iy = ixy % basis_.ny;
            const int ix = ixy / basis_.ny;
            const std::size_t index
                = static_cast<std::size_t>(reciprocal_index);
            gpu_box_indices_[index]
                = iz + iy * basis_.nz + ix * basis_.ny * basis_.nz;
            gpu_gx_[index] = basis_.gcar[reciprocal_index][0] * basis_.tpiba;
            gpu_gy_[index] = basis_.gcar[reciprocal_index][1] * basis_.tpiba;
            gpu_gz_[index] = basis_.gcar[reciprocal_index][2] * basis_.tpiba;
        }
    }
    if (use_gpu_)
    {
        gpu_workspace_.reset(new FdeGpuWorkspace());
        if (use_gpu_fft_)
        {
            gpu_workspace_->prepare_spectral(
                static_cast<std::size_t>(basis_.nx),
                static_cast<std::size_t>(basis_.ny),
                static_cast<std::size_t>(basis_.nz),
                gpu_box_indices_,
                gpu_gx_,
                gpu_gy_,
                gpu_gz_);
        }
    }
#else
    (void)use_gpu_;
#endif
}

PwGridDifferential::~PwGridDifferential() {}

bool PwGridDifferential::evaluate_kinetic_on_gpu(
    const std::vector<double>& density,
    const KineticFunctional functional,
    const double density_floor,
    const double volume_element,
    std::vector<double>& potential,
    double& energy_hartree) const
{
#ifdef __CUDA
    if (!use_gpu_ || gpu_workspace_.get() == nullptr
        || (functional != KineticFunctional::ThomasFermi && !use_gpu_fft_))
    {
        return false;
    }
    gpu_workspace_->kinetic_functional(density,
                                       functional,
                                       density_floor,
                                       volume_element,
                                       potential,
                                       energy_hartree);
    return true;
#else
    (void)density;
    (void)functional;
    (void)density_floor;
    (void)volume_element;
    (void)potential;
    (void)energy_hartree;
    return false;
#endif
}

std::size_t PwGridDifferential::local_size() const
{
    return static_cast<std::size_t>(basis_.nrxx);
}

bool PwGridDifferential::uses_gpu() const
{
#ifdef __CUDA
    return use_gpu_;
#else
    return false;
#endif
}

bool PwGridDifferential::all_processes(const bool local_condition) const
{
#ifdef __MPI
    if (basis_.pool_world == MPI_COMM_NULL)
    {
        throw std::runtime_error(
            "FDE PW grid agreement requires an initialized pool communicator");
    }
    const int local_value = local_condition ? 1 : 0;
    int global_value = 0;
    if (MPI_Allreduce(&local_value,
                      &global_value,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      basis_.pool_world)
        != MPI_SUCCESS)
    {
        throw std::runtime_error("FDE PW grid agreement reduction failed");
    }
    return global_value != 0;
#else
    return local_condition;
#endif
}

void PwGridDifferential::gradient(const std::vector<double>& values,
                                  std::vector<double>& gradient_x,
                                  std::vector<double>& gradient_y,
                                  std::vector<double>& gradient_z) const
{
    if (values.size() != this->local_size())
    {
        throw std::invalid_argument("FDE PW gradient input does not match local nrxx");
    }
#ifdef __CUDA
    if (use_gpu_fft_)
    {
        gpu_workspace_->spectral_gradient(values,
                                          gradient_x,
                                          gradient_y,
                                          gradient_z);
        return;
    }
#endif
    const std::complex<double> imaginary_unit(0.0, 1.0);
    std::vector<std::complex<double>> reciprocal(static_cast<std::size_t>(basis_.npw));
    std::vector<std::complex<double>> derivative(static_cast<std::size_t>(basis_.npw));
    basis_.real2recip(values.data(), reciprocal.data());

    std::vector<double>* components[3] = {&gradient_x, &gradient_y, &gradient_z};
    for (int direction = 0; direction < 3; ++direction)
    {
        components[direction]->assign(this->local_size(), 0.0);
#pragma omp parallel for schedule(static)
        for (int reciprocal_index = 0; reciprocal_index < basis_.npw; ++reciprocal_index)
        {
            derivative[static_cast<std::size_t>(reciprocal_index)]
                = imaginary_unit * reciprocal[static_cast<std::size_t>(reciprocal_index)]
                  * basis_.gcar[reciprocal_index][direction] * basis_.tpiba;
        }
        basis_.recip2real(derivative.data(), components[direction]->data());
    }
}

std::vector<double> PwGridDifferential::divergence(
    const std::vector<double>& vector_x,
    const std::vector<double>& vector_y,
    const std::vector<double>& vector_z) const
{
    if (vector_x.size() != this->local_size()
        || vector_y.size() != this->local_size()
        || vector_z.size() != this->local_size())
    {
        throw std::invalid_argument("FDE PW divergence input does not match local nrxx");
    }
#ifdef __CUDA
    if (use_gpu_fft_)
    {
        return gpu_workspace_->spectral_divergence(vector_x,
                                                   vector_y,
                                                   vector_z);
    }
#endif
    const std::complex<double> imaginary_unit(0.0, 1.0);
    std::vector<std::complex<double>> component_reciprocal(
        static_cast<std::size_t>(basis_.npw));
    std::vector<std::complex<double>> divergence_reciprocal(
        static_cast<std::size_t>(basis_.npw),
        std::complex<double>(0.0, 0.0));
    const std::vector<double>* components[3] = {&vector_x, &vector_y, &vector_z};
    for (int direction = 0; direction < 3; ++direction)
    {
        basis_.real2recip(components[direction]->data(), component_reciprocal.data());
#pragma omp parallel for schedule(static)
        for (int reciprocal_index = 0; reciprocal_index < basis_.npw; ++reciprocal_index)
        {
            divergence_reciprocal[static_cast<std::size_t>(reciprocal_index)]
                += imaginary_unit
                   * component_reciprocal[static_cast<std::size_t>(reciprocal_index)]
                   * basis_.gcar[reciprocal_index][direction] * basis_.tpiba;
        }
    }
    std::vector<double> result(this->local_size(), 0.0);
    basis_.recip2real(divergence_reciprocal.data(), result.data());
    return result;
}

} // namespace fde
