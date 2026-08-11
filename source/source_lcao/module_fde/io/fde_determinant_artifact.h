#ifndef FDE_DETERMINANT_ARTIFACT_H
#define FDE_DETERMINANT_ARTIFACT_H

#include "source_lcao/module_fde/embedding/fde_subspace_solver.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

struct OccupiedSpinOrbitals
{
    std::vector<double> coefficients;
    std::vector<double> orbital_energies_ry;
    std::vector<std::string> source_fragment_labels;
};

struct DiabaticDeterminantArtifact
{
    int schema_version;
    std::string state_label;
    std::string geometry_fingerprint;
    std::string orbital_fingerprint;
    std::size_t ao_dimension;
    OccupiedSpinOrbitals alpha;
    OccupiedSpinOrbitals beta;
};

struct FragmentOrbitalSolution
{
    std::string fragment_label;
    std::vector<std::size_t> active_orbitals;
    int alpha_electrons;
    int beta_electrons;
    ActiveSubspaceSolution alpha_solution;
    ActiveSubspaceSolution beta_solution;
};

struct DeterminantArtifactMetadata
{
    std::string state_label;
    std::string geometry_fingerprint;
    std::string orbital_fingerprint;
};

class DeterminantArtifactIO
{
  public:
    static DiabaticDeterminantArtifact build(
        const DeterminantArtifactMetadata& metadata,
        std::size_t full_ao_dimension,
        const std::vector<FragmentOrbitalSolution>& fragments,
        const std::vector<double>& full_overlap,
        double orthonormality_tolerance);

    static void validate(const DiabaticDeterminantArtifact& artifact,
                         const std::vector<double>& full_overlap,
                         double orthonormality_tolerance);

    static std::size_t occupied_count(const OccupiedSpinOrbitals& orbitals,
                                      std::size_t ao_dimension);

    static void write(std::ostream& output,
                      const DiabaticDeterminantArtifact& artifact,
                      const std::vector<double>& full_overlap,
                      double orthonormality_tolerance);

    static DiabaticDeterminantArtifact read(std::istream& input,
                                            const std::vector<double>& full_overlap,
                                            double orthonormality_tolerance);
};

} // namespace fde

#endif // FDE_DETERMINANT_ARTIFACT_H
