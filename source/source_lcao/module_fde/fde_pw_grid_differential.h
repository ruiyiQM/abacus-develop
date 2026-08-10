#ifndef FDE_PW_GRID_DIFFERENTIAL_H
#define FDE_PW_GRID_DIFFERENTIAL_H

#include "fde_semilocal_functional.h"

#include <vector>

namespace ModulePW
{
class PW_Basis;
}

namespace fde
{

/** Grid derivatives implemented by the distributed ABACUS PW FFT path. */
class PwGridDifferential : public GridDifferentialOperator
{
  public:
    explicit PwGridDifferential(const ModulePW::PW_Basis& basis,
                                bool use_gpu);

    std::size_t local_size() const override;
    bool uses_gpu() const override;
    void gradient(const std::vector<double>& values,
                  std::vector<double>& gradient_x,
                  std::vector<double>& gradient_y,
                  std::vector<double>& gradient_z) const override;
    std::vector<double> divergence(const std::vector<double>& vector_x,
                                   const std::vector<double>& vector_y,
                                   const std::vector<double>& vector_z) const override;

  private:
    const ModulePW::PW_Basis& basis_;
    bool use_gpu_;
    bool use_gpu_fft_;
    std::vector<int> gpu_box_indices_;
    std::vector<double> gpu_gx_;
    std::vector<double> gpu_gy_;
    std::vector<double> gpu_gz_;
};

} // namespace fde

#endif // FDE_PW_GRID_DIFFERENTIAL_H
