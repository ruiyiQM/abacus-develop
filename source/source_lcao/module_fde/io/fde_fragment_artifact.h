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
struct FragmentScfArtifact
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
    OccupiedSpinOrbitals alpha;
    OccupiedSpinOrbitals beta;
    std::vector<double> ao_overlap;
    std::vector<double> hamiltonian_alpha_ry;
    std::vector<double> hamiltonian_beta_ry;
    double subsystem_total_energy_ry;
    double ion_ion_energy_ry;
    double hartree_cross_energy_ry;
    double nonadditive_kinetic_energy_ry;
    double nonadditive_xc_energy_ry;
};

class FragmentScfArtifactIO
{
  public:
    static void validate(const FragmentScfArtifact& artifact,
                         double symmetry_tolerance);

    static void write(std::ostream& output,
                      const FragmentScfArtifact& artifact,
                      double symmetry_tolerance);

    static FragmentScfArtifact read(std::istream& input,
                                    double symmetry_tolerance);
};

} // namespace fde

#endif // FDE_FRAGMENT_ARTIFACT_H
