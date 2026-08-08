#include "fde_density_artifact.h"

#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace fde
{

namespace
{

const int artifact_schema_version = 1;

void require_token(std::istream& input, const char* expected)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        throw std::invalid_argument(std::string("FDE density artifact expected token ") + expected);
    }
}

void validate_text_field(const std::string& value, const char* name)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(std::string("FDE density artifact ") + name
                                    + " must be nonempty and contain no whitespace");
    }
}

double integrated_density(const std::vector<double>& density,
                          const double volume_element)
{
    double electrons = 0.0;
    for (std::size_t index = 0; index < density.size(); ++index)
    {
        if (!std::isfinite(density[index]) || density[index] < 0.0)
        {
            throw std::invalid_argument("FDE density artifact contains a negative or non-finite density");
        }
        electrons += density[index] * volume_element;
    }
    return electrons;
}

void require_equal(const std::string& first,
                   const std::string& second,
                   const char* description)
{
    if (first != second)
    {
        throw std::invalid_argument(std::string("FDE density artifacts have different ") + description);
    }
}

} // namespace

void DensityArtifactIO::validate(const FrozenDensityArtifact& artifact,
                                 const double electron_tolerance)
{
    if (artifact.schema_version != artifact_schema_version)
    {
        throw std::invalid_argument("Unsupported FDE density artifact schema version");
    }
    if (!std::isfinite(electron_tolerance) || electron_tolerance < 0.0)
    {
        throw std::invalid_argument("FDE electron tolerance must be finite and nonnegative");
    }

    validate_text_field(artifact.fragment_label, "fragment label");
    validate_text_field(artifact.state_label, "state label");
    validate_text_field(artifact.geometry_fingerprint, "geometry fingerprint");
    validate_text_field(artifact.grid_fingerprint, "grid fingerprint");
    validate_text_field(artifact.pseudopotential_fingerprint, "pseudopotential fingerprint");
    validate_text_field(artifact.orbital_fingerprint, "orbital fingerprint");
    validate_text_field(artifact.core_density_fingerprint, "core-density fingerprint");
    validate_text_field(artifact.xc_functional, "XC functional");
    validate_text_field(artifact.kinetic_functional, "kinetic functional");

    if (artifact.grid_x == 0 || artifact.grid_y == 0 || artifact.grid_z == 0)
    {
        throw std::invalid_argument("FDE density artifact grid dimensions must be positive");
    }
    if (artifact.grid_x > std::numeric_limits<std::size_t>::max() / artifact.grid_y
        || artifact.grid_x * artifact.grid_y > std::numeric_limits<std::size_t>::max() / artifact.grid_z)
    {
        throw std::overflow_error("FDE density artifact grid size overflows size_t");
    }
    const std::size_t grid_size = artifact.grid_x * artifact.grid_y * artifact.grid_z;
    if (artifact.rho_alpha_bohr3.size() != grid_size
        || artifact.rho_beta_bohr3.size() != grid_size)
    {
        throw std::invalid_argument("FDE density artifact arrays do not match the grid dimensions");
    }
    if (!std::isfinite(artifact.cell_volume_bohr3) || artifact.cell_volume_bohr3 <= 0.0)
    {
        throw std::invalid_argument("FDE density artifact cell volume must be finite and positive");
    }
    if (artifact.alpha_electrons < 0 || artifact.beta_electrons < 0
        || artifact.freeze_thaw_cycle < 0)
    {
        throw std::invalid_argument("FDE density artifact populations and cycle must be nonnegative");
    }
    if (!std::isfinite(artifact.orbital_kinetic_energy_ry)
        || !std::isfinite(artifact.nonlocal_pseudopotential_energy_ry))
    {
        throw std::invalid_argument("FDE density artifact energy terms must be finite");
    }

    const double volume_element = artifact.cell_volume_bohr3 / static_cast<double>(grid_size);
    const double alpha_integral = integrated_density(artifact.rho_alpha_bohr3, volume_element);
    const double beta_integral = integrated_density(artifact.rho_beta_bohr3, volume_element);
    if (std::fabs(alpha_integral - artifact.alpha_electrons) > electron_tolerance
        || std::fabs(beta_integral - artifact.beta_electrons) > electron_tolerance)
    {
        throw std::invalid_argument("FDE density artifact integrals do not reproduce spin populations");
    }
}

void DensityArtifactIO::validate_compatible_pair(const FrozenDensityArtifact& first,
                                                 const FrozenDensityArtifact& second,
                                                 const double electron_tolerance)
{
    DensityArtifactIO::validate(first, electron_tolerance);
    DensityArtifactIO::validate(second, electron_tolerance);
    if (first.fragment_label == second.fragment_label)
    {
        throw std::invalid_argument("FDE density pair must contain two different fragments");
    }
    require_equal(first.state_label, second.state_label, "state labels");
    require_equal(first.geometry_fingerprint, second.geometry_fingerprint, "geometries");
    require_equal(first.grid_fingerprint, second.grid_fingerprint, "grids");
    require_equal(first.pseudopotential_fingerprint,
                  second.pseudopotential_fingerprint,
                  "pseudopotentials");
    require_equal(first.orbital_fingerprint, second.orbital_fingerprint, "orbital bases");
    require_equal(first.xc_functional, second.xc_functional, "XC functionals");
    require_equal(first.kinetic_functional, second.kinetic_functional, "kinetic functionals");
    if (first.grid_x != second.grid_x || first.grid_y != second.grid_y
        || first.grid_z != second.grid_z || first.cell_volume_bohr3 != second.cell_volume_bohr3)
    {
        throw std::invalid_argument("FDE density artifacts have incompatible real-space grids");
    }
}

void DensityArtifactIO::write(std::ostream& output, const FrozenDensityArtifact& artifact)
{
    DensityArtifactIO::validate(artifact, 1.0e-8);
    output << std::setprecision(17);
    output << "FDE_DENSITY_ARTIFACT " << artifact.schema_version << '\n';
    output << "FRAGMENT " << artifact.fragment_label << '\n';
    output << "STATE " << artifact.state_label << '\n';
    output << "GEOMETRY " << artifact.geometry_fingerprint << '\n';
    output << "GRID_FINGERPRINT " << artifact.grid_fingerprint << '\n';
    output << "PSEUDOPOTENTIALS " << artifact.pseudopotential_fingerprint << '\n';
    output << "ORBITALS " << artifact.orbital_fingerprint << '\n';
    output << "CORE_DENSITY " << artifact.core_density_fingerprint << '\n';
    output << "FUNCTIONALS " << artifact.xc_functional << ' ' << artifact.kinetic_functional << '\n';
    output << "GRID " << artifact.grid_x << ' ' << artifact.grid_y << ' ' << artifact.grid_z
           << ' ' << artifact.cell_volume_bohr3 << '\n';
    output << "POPULATIONS " << artifact.alpha_electrons << ' ' << artifact.beta_electrons << '\n';
    output << "SCF " << artifact.freeze_thaw_cycle << ' ' << (artifact.scf_converged ? 1 : 0) << '\n';
    output << "ENERGIES_RY " << artifact.orbital_kinetic_energy_ry << ' '
           << artifact.nonlocal_pseudopotential_energy_ry << '\n';
    output << "RHO_ALPHA " << artifact.rho_alpha_bohr3.size();
    for (std::size_t index = 0; index < artifact.rho_alpha_bohr3.size(); ++index)
    {
        output << ' ' << artifact.rho_alpha_bohr3[index];
    }
    output << '\n';
    output << "RHO_BETA " << artifact.rho_beta_bohr3.size();
    for (std::size_t index = 0; index < artifact.rho_beta_bohr3.size(); ++index)
    {
        output << ' ' << artifact.rho_beta_bohr3[index];
    }
    output << "\nEND\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE density artifact");
    }
}

FrozenDensityArtifact DensityArtifactIO::read(std::istream& input,
                                              const double electron_tolerance)
{
    FrozenDensityArtifact artifact;
    require_token(input, "FDE_DENSITY_ARTIFACT");
    input >> artifact.schema_version;
    require_token(input, "FRAGMENT");
    input >> artifact.fragment_label;
    require_token(input, "STATE");
    input >> artifact.state_label;
    require_token(input, "GEOMETRY");
    input >> artifact.geometry_fingerprint;
    require_token(input, "GRID_FINGERPRINT");
    input >> artifact.grid_fingerprint;
    require_token(input, "PSEUDOPOTENTIALS");
    input >> artifact.pseudopotential_fingerprint;
    require_token(input, "ORBITALS");
    input >> artifact.orbital_fingerprint;
    require_token(input, "CORE_DENSITY");
    input >> artifact.core_density_fingerprint;
    require_token(input, "FUNCTIONALS");
    input >> artifact.xc_functional >> artifact.kinetic_functional;
    require_token(input, "GRID");
    input >> artifact.grid_x >> artifact.grid_y >> artifact.grid_z >> artifact.cell_volume_bohr3;
    require_token(input, "POPULATIONS");
    input >> artifact.alpha_electrons >> artifact.beta_electrons;
    require_token(input, "SCF");
    int converged = 0;
    input >> artifact.freeze_thaw_cycle >> converged;
    if (converged != 0 && converged != 1)
    {
        throw std::invalid_argument("FDE density artifact SCF flag must be zero or one");
    }
    artifact.scf_converged = converged == 1;
    require_token(input, "ENERGIES_RY");
    input >> artifact.orbital_kinetic_energy_ry >> artifact.nonlocal_pseudopotential_energy_ry;

    require_token(input, "RHO_ALPHA");
    std::size_t alpha_size = 0;
    input >> alpha_size;
    artifact.rho_alpha_bohr3.resize(alpha_size);
    for (std::size_t index = 0; index < alpha_size; ++index)
    {
        input >> artifact.rho_alpha_bohr3[index];
    }
    require_token(input, "RHO_BETA");
    std::size_t beta_size = 0;
    input >> beta_size;
    artifact.rho_beta_bohr3.resize(beta_size);
    for (std::size_t index = 0; index < beta_size; ++index)
    {
        input >> artifact.rho_beta_bohr3[index];
    }
    require_token(input, "END");
    if (!input)
    {
        throw std::invalid_argument("FDE density artifact is truncated or malformed");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE density artifact contains trailing content");
    }

    DensityArtifactIO::validate(artifact, electron_tolerance);
    return artifact;
}

} // namespace fde
