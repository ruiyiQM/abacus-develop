#ifndef FDE_IO_KPOINT_BAND_ARTIFACT_H
#define FDE_IO_KPOINT_BAND_ARTIFACT_H

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

struct FdeKPointBand
{
    int spin;
    std::size_t physical_kpoint;
    double kx_direct;
    double ky_direct;
    double kz_direct;
    double weight;
    std::vector<double> eigenvalues_ry;
};

/**
 * Solver-independent k-resolved eigenvalue record for one converged embedded
 * fragment SCF calculation.  It deliberately contains no determinant data:
 * periodic FODFT coupling requires a separate Born-von-Karman convention.
 */
struct FdeKPointBandArtifact
{
    int schema_version;
    std::string state_label;
    std::string fragment_label;
    std::string geometry_fingerprint;
    std::string orbital_fingerprint;
    std::size_t ao_dimension;
    std::size_t band_count;
    std::vector<FdeKPointBand> points;
};

class FdeKPointBandArtifactIO
{
  public:
    static void validate(const FdeKPointBandArtifact& artifact,
                         double tolerance);

    static void write(std::ostream& output,
                      const FdeKPointBandArtifact& artifact,
                      double tolerance);

    static FdeKPointBandArtifact read(std::istream& input,
                                      double tolerance);
};

} // namespace fde

#endif // FDE_IO_KPOINT_BAND_ARTIFACT_H
