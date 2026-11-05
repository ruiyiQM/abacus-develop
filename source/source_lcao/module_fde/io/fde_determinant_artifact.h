#ifndef FDE_DETERMINANT_ARTIFACT_H
#define FDE_DETERMINANT_ARTIFACT_H

#include "source_lcao/module_fde/embedding/fde_subspace_solver.h"

#include <cstddef>
#include <complex>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

inline double conjugate_scalar(const double value)
{
    return value;
}

inline std::complex<double> conjugate_scalar(const std::complex<double>& value)
{
    return std::conj(value);
}

void validate_finite_state_boundary(int schema_version,
                                    const std::vector<double>& boundary_twist,
                                    bool complex_scalar);

void report_artifact_file(const std::string& path,
                          std::ostream& running_log,
                          bool reading);

template <typename Scalar>
struct ScalarOccupiedSpinOrbitals
{
    std::vector<Scalar> coefficients;
    std::vector<double> orbital_energies_ry;
    std::vector<std::string> source_fragment_labels;
};

using OccupiedSpinOrbitals = ScalarOccupiedSpinOrbitals<double>;
using ComplexOccupiedSpinOrbitals = ScalarOccupiedSpinOrbitals<std::complex<double>>;

template <typename Scalar>
struct ScalarDiabaticDeterminantArtifact
{
    int schema_version;
    std::string state_label;
    std::string geometry_fingerprint;
    std::string orbital_fingerprint;
    std::size_t ao_dimension;
    ScalarOccupiedSpinOrbitals<Scalar> alpha;
    ScalarOccupiedSpinOrbitals<Scalar> beta;
    // V2 complex artifacts use one canonical reciprocal-cell twist for the whole finite state.
    std::vector<double> boundary_twist;
    std::string resource_fingerprint;
};

using DiabaticDeterminantArtifact = ScalarDiabaticDeterminantArtifact<double>;
using ComplexDiabaticDeterminantArtifact
    = ScalarDiabaticDeterminantArtifact<std::complex<double>>;

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

template <typename Scalar>
class ScalarDeterminantArtifactIO
{
  public:
    static void validate(const ScalarDiabaticDeterminantArtifact<Scalar>& artifact,
                         const std::vector<Scalar>& full_overlap,
                         double orthonormality_tolerance);

    static std::size_t occupied_count(const ScalarOccupiedSpinOrbitals<Scalar>& orbitals,
                                      std::size_t ao_dimension);

    static void write(std::ostream& output,
                      const ScalarDiabaticDeterminantArtifact<Scalar>& artifact,
                      const std::vector<Scalar>& full_overlap,
                      double orthonormality_tolerance);

    static ScalarDiabaticDeterminantArtifact<Scalar> read(
        std::istream& input,
        const std::vector<Scalar>& full_overlap,
        double orthonormality_tolerance);
};

class DeterminantArtifactIO : public ScalarDeterminantArtifactIO<double>
{
  public:
    static DiabaticDeterminantArtifact build(
        const DeterminantArtifactMetadata& metadata,
        std::size_t full_ao_dimension,
        const std::vector<FragmentOrbitalSolution>& fragments,
        const std::vector<double>& full_overlap,
        double orthonormality_tolerance);
};

using ComplexDeterminantArtifactIO = ScalarDeterminantArtifactIO<std::complex<double>>;

} // namespace fde

#endif // FDE_DETERMINANT_ARTIFACT_H
