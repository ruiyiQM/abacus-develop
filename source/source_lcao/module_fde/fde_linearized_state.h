#ifndef FDE_LINEARIZED_STATE_H
#define FDE_LINEARIZED_STATE_H

#include "fde_electronic_coupling.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

struct LinearizedStateArtifact
{
    int schema_version;
    std::string state_label;
    std::string geometry_fingerprint;
    std::size_t ao_dimension;
    double reference_energy_ry;
    std::vector<double> hamiltonian_alpha_ry;
    std::vector<double> hamiltonian_beta_ry;
};

class LinearizedStateArtifactIO
{
  public:
    static void validate(const LinearizedStateArtifact& artifact,
                         double symmetry_tolerance);
    static void write(std::ostream& output,
                      const LinearizedStateArtifact& artifact,
                      double symmetry_tolerance);
    static LinearizedStateArtifact read(std::istream& input,
                                        double symmetry_tolerance);
};

/** First-order state-specific DFT energy model for a transition density. */
class LinearizedTransitionEnergy : public TransitionEnergyProvider
{
  public:
    LinearizedTransitionEnergy(const std::vector<LinearizedStateArtifact>& states,
                               const std::vector<double>& ao_overlap,
                               double singular_value_tolerance);

    double evaluate_ry(const DiabaticDeterminantArtifact& bra,
                       const DiabaticDeterminantArtifact& ket,
                       const SpinTransitionDensityMatrix& transition_density,
                       std::size_t ao_dimension) const override;

  private:
    std::vector<LinearizedStateArtifact> states_;
    std::vector<double> ao_overlap_;
    double singular_value_tolerance_;
};

} // namespace fde

#endif // FDE_LINEARIZED_STATE_H
