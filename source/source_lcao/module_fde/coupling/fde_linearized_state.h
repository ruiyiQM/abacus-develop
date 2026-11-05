#ifndef FDE_LINEARIZED_STATE_H
#define FDE_LINEARIZED_STATE_H

#include "source_lcao/module_fde/coupling/fde_electronic_coupling.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

template <typename Scalar>
struct ScalarLinearizedStateArtifact
{
    int schema_version;
    std::string state_label;
    std::string geometry_fingerprint;
    std::size_t ao_dimension;
    double reference_energy_ry;
    std::vector<Scalar> hamiltonian_alpha_ry;
    std::vector<Scalar> hamiltonian_beta_ry;
    std::string orbital_fingerprint;
    std::vector<double> boundary_twist;
};

template <typename Scalar>
class ScalarLinearizedStateArtifactIO
{
  public:
    static void validate(const ScalarLinearizedStateArtifact<Scalar>& artifact,
                         double symmetry_tolerance);
    static void write(std::ostream& output,
                      const ScalarLinearizedStateArtifact<Scalar>& artifact,
                      double symmetry_tolerance);
    static ScalarLinearizedStateArtifact<Scalar> read(std::istream& input,
                                        double symmetry_tolerance);
};

/** First-order state-specific DFT energy model for a transition density. */
template <typename Scalar>
class ScalarLinearizedTransitionEnergy : public ScalarTransitionEnergyProvider<Scalar>
{
  public:
    ScalarLinearizedTransitionEnergy(const std::vector<ScalarLinearizedStateArtifact<Scalar>>& states,
                               const std::vector<Scalar>& ao_overlap,
                               double singular_value_tolerance);

    std::string name() const override;

    Scalar evaluate_ry(const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
                       const ScalarDiabaticDeterminantArtifact<Scalar>& ket,
                       const ScalarSpinTransitionDensityMatrix<Scalar>& transition_density,
                       std::size_t ao_dimension) const override;

  private:
    std::vector<ScalarLinearizedStateArtifact<Scalar>> states_;
    std::vector<Scalar> ao_overlap_;
    double singular_value_tolerance_;
};

using LinearizedStateArtifact = ScalarLinearizedStateArtifact<double>;
using ComplexLinearizedStateArtifact = ScalarLinearizedStateArtifact<std::complex<double>>;
using LinearizedStateArtifactIO = ScalarLinearizedStateArtifactIO<double>;
using ComplexLinearizedStateArtifactIO = ScalarLinearizedStateArtifactIO<std::complex<double>>;
using LinearizedTransitionEnergy = ScalarLinearizedTransitionEnergy<double>;
using ComplexLinearizedTransitionEnergy = ScalarLinearizedTransitionEnergy<std::complex<double>>;

} // namespace fde

#endif // FDE_LINEARIZED_STATE_H
