#include <complex>
#include "pw_basis_k.h"
#include <cassert>
#include "../module_base/timer.h"
#include "pw_gatherscatter.h"
#include "module_pw/kernels/pw_op.h"

namespace ModulePW
{

///
/// transform real space to reciprocal space
/// in: (nplane, ny, nx), complex<double> data
/// out: (nz, ns),  complex<double> data
///
template <typename FPTYPE>
void PW_Basis_K:: real2recip(const std::complex<FPTYPE> * in, std::complex<FPTYPE> * out, const int ik, const bool add, const FPTYPE factor)
{
    ModuleBase::timer::tick(this->classname, "real2recip");

    assert(this->gamma_only == false);
    for(int ir = 0 ; ir < this->nrxx ; ++ir)
    {
        this->ft.get_auxr_data<FPTYPE>()[ir] = in[ir];
    }
    this->ft.fftxyfor(ft.get_auxr_data<FPTYPE>(),ft.get_auxr_data<FPTYPE>());

    this->gatherp_scatters(this->ft.get_auxr_data<FPTYPE>(), this->ft.get_auxg_data<FPTYPE>());
    
    this->ft.fftzfor(ft.get_auxg_data<FPTYPE>(), ft.get_auxg_data<FPTYPE>());

    const int startig = ik*this->npwk_max;
    const int npwk = this->npwk[ik];
    if(add) {
        for (int igl = 0; igl < npwk; ++igl) {
            out[igl] += factor / FPTYPE(this->nxyz) * this->ft.get_auxg_data<FPTYPE>()[this->igl2isz_k[igl + startig]];
        }
    }
    else {
        for (int igl = 0; igl < npwk; ++igl) {
            out[igl] = this->ft.get_auxg_data<FPTYPE>()[this->igl2isz_k[igl + startig]] / FPTYPE(this->nxyz);
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
}

///
/// transform real space to reciprocal space
/// in: (nplane, ny, nx), double data
/// out: (nz, ns), complex<double> data
///
template <typename FPTYPE>
void PW_Basis_K:: real2recip(const FPTYPE * in, std::complex<FPTYPE> * out, const int ik, const bool add, const FPTYPE factor)
{
    ModuleBase::timer::tick(this->classname, "real2recip");
    assert(this->gamma_only == true);
    // for(int ir = 0 ; ir < this->nrxx ; ++ir)
    // {
    //     this->ft.get_rspace_data<FPTYPE>()[ir] = in[ir];
    // }
    // r2c in place
    const int npy = this->ny * this->nplane;
    for(int ix = 0 ; ix < this->nx ; ++ix)
    {
        const int ixpy = ix*npy;
        for(int ipy = 0 ; ipy < npy ; ++ipy)
        {
            this->ft.get_rspace_data<FPTYPE>()[ixpy + ipy] = in[ixpy + ipy];
        }
    }

    this->ft.fftxyr2c(ft.get_rspace_data<FPTYPE>(),ft.get_auxr_data<FPTYPE>());

    this->gatherp_scatters(this->ft.get_auxr_data<FPTYPE>(), this->ft.get_auxg_data<FPTYPE>());
    
    this->ft.fftzfor(ft.get_auxg_data<FPTYPE>(),ft.get_auxg_data<FPTYPE>());

    const int startig = ik*this->npwk_max;
    const int npwk = this->npwk[ik];
    if(add)
    for(int igl = 0 ; igl < npwk ; ++igl)
    {
        out[igl] += factor / FPTYPE(this->nxyz) * this->ft.get_auxg_data<FPTYPE>()[this->igl2isz_k[igl+startig]];
    }
    else
    for(int igl = 0 ; igl < npwk ; ++igl)
    {
        out[igl] = this->ft.get_auxg_data<FPTYPE>()[this->igl2isz_k[igl+startig]] / FPTYPE(this->nxyz);
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
    return;
}

///
/// transform reciprocal space to real space
/// in: (nz,ns), complex<double>
/// out: (nplane, ny, nx), complex<double>
///
template <typename FPTYPE>
void PW_Basis_K:: recip2real(const std::complex<FPTYPE> * in, std::complex<FPTYPE> * out, const int ik, const bool add, const FPTYPE factor)
{
    ModuleBase::timer::tick(this->classname, "recip2real");
    assert(this->gamma_only == false);
    ModuleBase::GlobalFunc::ZEROS(ft.get_auxg_data<FPTYPE>(), this->nst * this->nz);

    const int startig = ik*this->npwk_max;
    const int npwk = this->npwk[ik];
    for(int igl = 0 ; igl < npwk ; ++igl)
    {
        this->ft.get_auxg_data<FPTYPE>()[this->igl2isz_k[igl+startig]] = in[igl];
    }
    this->ft.fftzbac(ft.get_auxg_data<FPTYPE>(), ft.get_auxg_data<FPTYPE>());

    this->gathers_scatterp(this->ft.get_auxg_data<FPTYPE>(),this->ft.get_auxr_data<FPTYPE>());

    this->ft.fftxybac(ft.get_auxr_data<FPTYPE>(),ft.get_auxr_data<FPTYPE>());
    
    if(add) {
        for (int ir = 0; ir < this->nrxx; ++ir) {
            out[ir] += factor * this->ft.get_auxr_data<FPTYPE>()[ir];
        }
    }
    else {
        for (int ir = 0; ir < this->nrxx; ++ir) {
            out[ir] = this->ft.get_auxr_data<FPTYPE>()[ir];
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

///
/// transform reciprocal space to real space
/// in: (nz,ns), complex<double>
/// out: (nplane, ny, nx), double
///
template <typename FPTYPE>
void PW_Basis_K:: recip2real(const std::complex<FPTYPE> * in, FPTYPE * out, const int ik, const bool add, const FPTYPE factor)
{
    ModuleBase::timer::tick(this->classname, "recip2real");
    assert(this->gamma_only == true);
    ModuleBase::GlobalFunc::ZEROS(ft.get_auxg_data<FPTYPE>(), this->nst * this->nz);

    const int startig = ik*this->npwk_max;
    const int npwk = this->npwk[ik];
    for(int igl = 0 ; igl < npwk ; ++igl)
    {
        this->ft.get_auxg_data<FPTYPE>()[this->igl2isz_k[igl+startig]] = in[igl];
    }
    this->ft.fftzbac(ft.get_auxg_data<FPTYPE>(), ft.get_auxg_data<FPTYPE>());
    
    this->gathers_scatterp(this->ft.get_auxg_data<FPTYPE>(), this->ft.get_auxr_data<FPTYPE>());

    this->ft.fftxyc2r(ft.get_auxr_data<FPTYPE>(),ft.get_rspace_data<FPTYPE>());

    // for(int ir = 0 ; ir < this->nrxx ; ++ir)
    // {
    //     out[ir] = this->ft.get_rspace_data<FPTYPE>()[ir] / this->nxyz;
    // }

    // r2c in place
    const int npy = this->ny * this->nplane;
    if(add) {
        for (int ix = 0; ix < this->nx; ++ix) {
            const int ixpy = ix * npy;
            for (int ipy = 0; ipy < npy; ++ipy) {
                out[ixpy + ipy] += factor * this->ft.get_rspace_data<FPTYPE>()[ixpy + ipy];
            }
        }
    }
    else {
        for (int ix = 0; ix < this->nx; ++ix) {
            const int ixpy = ix * npy;
            for (int ipy = 0; ipy < npy; ++ipy) {
                out[ixpy + ipy] = this->ft.get_rspace_data<FPTYPE>()[ixpy + ipy];
            }
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

template <>
void PW_Basis_K::real_to_recip(const psi::DEVICE_CPU * /*dev*/, const std::complex<float> * in, std::complex<float> * out, const int ik, const bool add, const float factor)
{
    this->real2recip(in, out, ik, add, factor);
}
template <>
void PW_Basis_K::real_to_recip(const psi::DEVICE_CPU * /*dev*/, const std::complex<double> * in, std::complex<double> * out, const int ik, const bool add, const double factor)
{
    this->real2recip(in, out, ik, add, factor);
}

template <>
void PW_Basis_K::recip_to_real(const psi::DEVICE_CPU * /*dev*/, const std::complex<float> * in, std::complex<float> * out, const int ik, const bool add, const float factor)
{
    this->recip2real(in, out, ik, add, factor);
}
template <>
void PW_Basis_K::recip_to_real(const psi::DEVICE_CPU * /*dev*/, const std::complex<double> * in, std::complex<double> * out, const int ik, const bool add, const double factor)
{
    this->recip2real(in, out, ik, add, factor);
}

#if (defined(__CUDA) || defined(__ROCM))
template <>
void PW_Basis_K::real_to_recip(const psi::DEVICE_GPU * ctx, const std::complex<float> * in, std::complex<float> * out, const int ik, const bool add, const float factor)
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

     psi::memory::synchronize_memory_op<std::complex<float>, psi::DEVICE_GPU, psi::DEVICE_GPU>()(
         ctx, ctx,
         this->ft.get_auxr_3d_data<float>(), in,
         this->nrxx);

     this->ft.fft3D_forward(ctx, this->ft.get_auxr_3d_data<float>(), this->ft.get_auxr_3d_data<float>());

    const int startig = ik*this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<float, psi::DEVICE_GPU>()(
        ctx, npw_k, this->nxyz, add, factor,  this->ig2ixyz_k + startig, this->ft.get_auxr_3d_data<float>(), out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}
template <>
void PW_Basis_K::real_to_recip(const psi::DEVICE_GPU * ctx, const std::complex<double> * in, std::complex<double> * out, const int ik, const bool add, const double factor)
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

     psi::memory::synchronize_memory_op<std::complex<double>, psi::DEVICE_GPU, psi::DEVICE_GPU>()(
         ctx, ctx,
         this->ft.get_auxr_3d_data<double>(), in,
         this->nrxx);

     this->ft.fft3D_forward(ctx, this->ft.get_auxr_3d_data<double>(), this->ft.get_auxr_3d_data<double>());

    const int startig = ik*this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<double, psi::DEVICE_GPU>()(
        ctx, npw_k, this->nxyz, add, factor,  this->ig2ixyz_k + startig, this->ft.get_auxr_3d_data<double>(), out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}

template <>
void PW_Basis_K::recip_to_real(const psi::DEVICE_GPU * ctx, const std::complex<float> * in, std::complex<float> * out, const int ik, const bool add, const float factor)
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(ft.get_auxr_3d_data<float>(), this->nxyz);
    psi::memory::set_memory_op<std::complex<float>, psi::DEVICE_GPU>()(
        ctx, this->ft.get_auxr_3d_data<float>(), 0, this->nxyz);

    const int startig = ik*this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<float, psi::DEVICE_GPU>()(
        ctx, npw_k, this->ig2ixyz_k + startig, in, this->ft.get_auxr_3d_data<float>());
    this->ft.fft3D_backward(ctx, this->ft.get_auxr_3d_data<float>(), this->ft.get_auxr_3d_data<float>());

    set_recip_to_real_output_op<float, psi::DEVICE_GPU>()(
        ctx, this->nrxx, add, factor, this->ft.get_auxr_3d_data<float>(), out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}
template <>
void PW_Basis_K::recip_to_real(const psi::DEVICE_GPU * ctx, const std::complex<double> * in, std::complex<double> * out, const int ik, const bool add, const double factor)
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(ft.get_auxr_3d_data<double>(), this->nxyz);
    psi::memory::set_memory_op<std::complex<double>, psi::DEVICE_GPU>()(
        ctx, this->ft.get_auxr_3d_data<double>(), 0, this->nxyz);

    const int startig = ik*this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<double, psi::DEVICE_GPU>()(
        ctx, npw_k, this->ig2ixyz_k + startig, in, this->ft.get_auxr_3d_data<double>());
    this->ft.fft3D_backward(ctx, this->ft.get_auxr_3d_data<double>(), this->ft.get_auxr_3d_data<double>());

    set_recip_to_real_output_op<double, psi::DEVICE_GPU>()(
        ctx, this->nrxx, add, factor, this->ft.get_auxr_3d_data<double>(), out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}
#endif

template void PW_Basis_K::real2recip<float>(const float* in, std::complex<float>* out, const int ik, const bool add = false, const float factor = 1.0); //in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::real2recip<float>(const std::complex<float>* in, std::complex<float>* out, const int ik, const bool add = false, const float factor = 1.0); //in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::recip2real<float>(const std::complex<float>* in, float* out, const int ik, const bool add = false, const float factor = 1.0); //in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis_K::recip2real<float>(const std::complex<float>* in, std::complex<float> * out, const int ik, const bool add = false, const float factor = 1.0); //in:(nz, ns)  ; out(nplane,nx*ny)

template void PW_Basis_K::real2recip<double>(const double* in, std::complex<double>* out, const int ik, const bool add = false, const double factor = 1.0); //in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::real2recip<double>(const std::complex<double>* in, std::complex<double>* out, const int ik, const bool add = false, const double factor = 1.0); //in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::recip2real<double>(const std::complex<double>* in, double* out, const int ik, const bool add = false, const double factor = 1.0); //in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis_K::recip2real<double>(const std::complex<double>* in, std::complex<double> * out, const int ik, const bool add = false, const double factor = 1.0); //in:(nz, ns)  ; out(nplane,nx*ny)
}