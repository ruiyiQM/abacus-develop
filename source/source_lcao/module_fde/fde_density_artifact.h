#ifndef FDE_DENSITY_ARTIFACT_H
#define FDE_DENSITY_ARTIFACT_H

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

struct FrozenDensityArtifact
{
    int schema_version;
    std::string fragment_label;
    std::string state_label;
    std::string geometry_fingerprint;
    std::string grid_fingerprint;
    std::string pseudopotential_fingerprint;
    std::string orbital_fingerprint;
    std::string core_density_fingerprint;
    std::string xc_functional;
    std::string kinetic_functional;
    std::size_t grid_x;
    std::size_t grid_y;
    std::size_t grid_z;
    double cell_volume_bohr3;
    int alpha_electrons;
    int beta_electrons;
    int freeze_thaw_cycle;
    bool scf_converged;
    double orbital_kinetic_energy_ry;
    double nonlocal_pseudopotential_energy_ry;
    std::vector<double> rho_alpha_bohr3;
    std::vector<double> rho_beta_bohr3;
};

class DensityArtifactIO
{
  public:
    static void validate(const FrozenDensityArtifact& artifact,
                         const double electron_tolerance);

    static void validate_compatible_pair(const FrozenDensityArtifact& first,
                                         const FrozenDensityArtifact& second,
                                         const double electron_tolerance);

    static void validate_compatible_set(const std::vector<FrozenDensityArtifact>& artifacts,
                                        const double electron_tolerance);

    static void write(std::ostream& output, const FrozenDensityArtifact& artifact);

    static FrozenDensityArtifact read(std::istream& input,
                                      const double electron_tolerance);
};

} // namespace fde

#endif // FDE_DENSITY_ARTIFACT_H
