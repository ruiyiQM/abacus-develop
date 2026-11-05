#ifndef FDE_FRAGMENT_ARTIFACT_H
#define FDE_FRAGMENT_ARTIFACT_H

#include "source_lcao/module_fde/io/fde_determinant_artifact.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

/**
 * Restart-independent result of one converged active-fragment SCF job.
 *
 * Matrices use the authoritative supersystem AO order and column-major
 * storage.  The Hamiltonian is the unprojected ABACUS Hamiltonian evaluated
 * at the final active density; the occupied columns retain zero coefficients
 * outside the active fragment AO space.
 */
template <typename Scalar>
struct ScalarFragmentScfArtifact
{
    int schema_version;
    std::string state_label;
    std::string fragment_label;
    std::string geometry_fingerprint;
    std::string orbital_fingerprint;
    std::string density_path;
    int freeze_thaw_cycle;
    bool scf_converged;
    std::size_t ao_dimension;
    std::vector<std::size_t> active_orbitals;
    ScalarOccupiedSpinOrbitals<Scalar> alpha;
    ScalarOccupiedSpinOrbitals<Scalar> beta;
    std::vector<Scalar> ao_overlap;
    std::vector<Scalar> hamiltonian_alpha_ry;
    std::vector<Scalar> hamiltonian_beta_ry;
    std::vector<double> boundary_twist;
    double subsystem_total_energy_ry;
    double ion_ion_energy_ry;
    double hartree_cross_energy_ry;
    double nonadditive_kinetic_energy_ry;
    double nonadditive_xc_energy_ry;
};

template <typename Scalar>
class ScalarFragmentScfArtifactIO
{
  public:
    static void validate(const ScalarFragmentScfArtifact<Scalar>& artifact,
                         double symmetry_tolerance);

    static void write(std::ostream& output,
                      const ScalarFragmentScfArtifact<Scalar>& artifact,
                      double symmetry_tolerance);

    static ScalarFragmentScfArtifact<Scalar> read(std::istream& input,
                                    double symmetry_tolerance);
};

using FragmentScfArtifact = ScalarFragmentScfArtifact<double>;
using ComplexFragmentScfArtifact = ScalarFragmentScfArtifact<std::complex<double>>;
using FragmentScfArtifactIO = ScalarFragmentScfArtifactIO<double>;
using ComplexFragmentScfArtifactIO = ScalarFragmentScfArtifactIO<std::complex<double>>;

} // namespace fde

#endif // FDE_FRAGMENT_ARTIFACT_H
