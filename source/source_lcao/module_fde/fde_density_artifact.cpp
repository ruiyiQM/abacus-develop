#include "fde_density_artifact.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <set>
#include <stdexcept>

namespace fde
{

namespace
{

const int current_artifact_schema_version = 2;
const int current_binary_format_version = 1;

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

void write_metadata(std::ostream& output,
                    const FrozenDensityArtifact& artifact)
{
    output << "FRAGMENT " << artifact.fragment_label << '\n';
    output << "STATE " << artifact.state_label << '\n';
    output << "GEOMETRY " << artifact.geometry_fingerprint << '\n';
    output << "GRID_FINGERPRINT " << artifact.grid_fingerprint << '\n';
    output << "PSEUDOPOTENTIALS " << artifact.pseudopotential_fingerprint << '\n';
    output << "ORBITALS " << artifact.orbital_fingerprint << '\n';
    output << "CORE_DENSITY " << artifact.core_density_fingerprint << '\n';
    output << "FUNCTIONALS " << artifact.xc_functional << ' '
           << artifact.kinetic_functional << '\n';
    output << "GRID " << artifact.grid_x << ' ' << artifact.grid_y << ' '
           << artifact.grid_z << ' ' << artifact.cell_volume_bohr3 << '\n';
    output << "POPULATIONS " << artifact.alpha_electrons << ' '
           << artifact.beta_electrons << '\n';
    output << "SCF " << artifact.freeze_thaw_cycle << ' '
           << (artifact.scf_converged ? 1 : 0);
    if (artifact.schema_version >= 2)
    {
        output << ' ' << artifact.scf_iterations << ' '
               << artifact.scf_density_residual;
    }
    output << '\n';
    output << "ENERGIES_RY " << artifact.orbital_kinetic_energy_ry << ' '
           << artifact.nonlocal_pseudopotential_energy_ry << '\n';
}

void read_metadata(std::istream& input,
                   FrozenDensityArtifact& artifact)
{
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
    input >> artifact.grid_x >> artifact.grid_y >> artifact.grid_z
          >> artifact.cell_volume_bohr3;
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
    artifact.scf_iterations = 0;
    artifact.scf_density_residual = 0.0;
    if (artifact.schema_version >= 2)
    {
        input >> artifact.scf_iterations >> artifact.scf_density_residual;
    }
    require_token(input, "ENERGIES_RY");
    input >> artifact.orbital_kinetic_energy_ry
          >> artifact.nonlocal_pseudopotential_energy_ry;
}

std::size_t artifact_grid_size(const FrozenDensityArtifact& artifact)
{
    if (artifact.grid_x == 0 || artifact.grid_y == 0 || artifact.grid_z == 0
        || artifact.grid_x > std::numeric_limits<std::size_t>::max() / artifact.grid_y
        || artifact.grid_x * artifact.grid_y
               > std::numeric_limits<std::size_t>::max() / artifact.grid_z)
    {
        throw std::invalid_argument("FDE density artifact grid is invalid or overflows");
    }
    return artifact.grid_x * artifact.grid_y * artifact.grid_z;
}

bool host_is_little_endian()
{
    const std::uint16_t value = 1;
    unsigned char first = 0;
    std::memcpy(&first, &value, sizeof(first));
    return first == 1;
}

std::uint64_t reverse_bytes(std::uint64_t value)
{
    value = ((value & UINT64_C(0x00ff00ff00ff00ff)) << 8)
            | ((value & UINT64_C(0xff00ff00ff00ff00)) >> 8);
    value = ((value & UINT64_C(0x0000ffff0000ffff)) << 16)
            | ((value & UINT64_C(0xffff0000ffff0000)) >> 16);
    return (value << 32) | (value >> 32);
}

void write_bytes(std::ostream& output,
                 const char* data,
                 const std::size_t size)
{
    std::size_t offset = 0;
    const std::size_t maximum_chunk
        = static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
    while (offset < size)
    {
        const std::size_t chunk = std::min(size - offset, maximum_chunk);
        output.write(data + offset, static_cast<std::streamsize>(chunk));
        offset += chunk;
    }
}

void read_bytes(std::istream& input,
                char* data,
                const std::size_t size)
{
    std::size_t offset = 0;
    const std::size_t maximum_chunk
        = static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
    while (offset < size)
    {
        const std::size_t chunk = std::min(size - offset, maximum_chunk);
        input.read(data + offset, static_cast<std::streamsize>(chunk));
        if (!input)
        {
            throw std::invalid_argument("FDE binary density artifact is truncated");
        }
        offset += chunk;
    }
}

void write_binary_density(std::ostream& output,
                          const std::vector<double>& density)
{
    if (density.size() > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::overflow_error("FDE binary density byte count overflows size_t");
    }
    if (host_is_little_endian())
    {
        write_bytes(output,
                    reinterpret_cast<const char*>(density.data()),
                    density.size() * sizeof(double));
        return;
    }
    std::vector<std::uint64_t> buffer(std::min<std::size_t>(density.size(), 4096));
    std::size_t offset = 0;
    while (offset < density.size())
    {
        const std::size_t count = std::min(buffer.size(), density.size() - offset);
        for (std::size_t index = 0; index < count; ++index)
        {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &density[offset + index], sizeof(bits));
            buffer[index] = reverse_bytes(bits);
        }
        write_bytes(output,
                    reinterpret_cast<const char*>(buffer.data()),
                    count * sizeof(std::uint64_t));
        offset += count;
    }
}

void read_binary_density(std::istream& input,
                         const std::size_t size,
                         std::vector<double>& density)
{
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::overflow_error("FDE binary density byte count overflows size_t");
    }
    density.resize(size);
    read_bytes(input,
               reinterpret_cast<char*>(density.data()),
               size * sizeof(double));
    if (!host_is_little_endian())
    {
        for (std::size_t index = 0; index < density.size(); ++index)
        {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &density[index], sizeof(bits));
            bits = reverse_bytes(bits);
            std::memcpy(&density[index], &bits, sizeof(bits));
        }
    }
}

void require_line_end(std::istream& input)
{
    char character = 0;
    do
    {
        if (!input.get(character))
        {
            throw std::invalid_argument("FDE binary density artifact is truncated");
        }
    } while (character == ' ' || character == '\t');
    if (character == '\r')
    {
        if (!input.get(character) || character != '\n')
        {
            throw std::invalid_argument("FDE binary density artifact has an invalid line ending");
        }
    }
    else if (character != '\n')
    {
        throw std::invalid_argument("FDE binary density artifact has an invalid line ending");
    }
}

void require_no_trailing_content(std::istream& input,
                                 const char* description)
{
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument(std::string(description) + " contains trailing content");
    }
}

FrozenDensityArtifact read_text_body(std::istream& input,
                                     const int schema_version,
                                     const double electron_tolerance)
{
    FrozenDensityArtifact artifact;
    artifact.schema_version = schema_version;
    read_metadata(input, artifact);
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
    require_no_trailing_content(input, "FDE density artifact");
    DensityArtifactIO::validate(artifact, electron_tolerance);
    return artifact;
}

FrozenDensityArtifact read_binary_body(std::istream& input,
                                       const int format_version,
                                       const int schema_version,
                                       const double electron_tolerance)
{
    if (format_version != current_binary_format_version)
    {
        throw std::invalid_argument("Unsupported FDE binary density format version");
    }
    FrozenDensityArtifact artifact;
    artifact.schema_version = schema_version;
    read_metadata(input, artifact);
    require_token(input, "BYTE_ORDER");
    std::string byte_order;
    input >> byte_order;
    if (byte_order != "LITTLE_ENDIAN")
    {
        throw std::invalid_argument("Unsupported FDE binary density byte order");
    }
    const std::size_t expected_size = artifact_grid_size(artifact);
    require_token(input, "RHO_ALPHA_BINARY");
    std::size_t alpha_size = 0;
    input >> alpha_size;
    if (alpha_size != expected_size)
    {
        throw std::invalid_argument("FDE binary alpha density size does not match the grid");
    }
    require_line_end(input);
    read_binary_density(input, alpha_size, artifact.rho_alpha_bohr3);
    require_line_end(input);
    require_token(input, "RHO_BETA_BINARY");
    std::size_t beta_size = 0;
    input >> beta_size;
    if (beta_size != expected_size)
    {
        throw std::invalid_argument("FDE binary beta density size does not match the grid");
    }
    require_line_end(input);
    read_binary_density(input, beta_size, artifact.rho_beta_bohr3);
    require_line_end(input);
    require_token(input, "END");
    require_no_trailing_content(input, "FDE binary density artifact");
    DensityArtifactIO::validate(artifact, electron_tolerance);
    return artifact;
}

FrozenDensityArtifact read_seed_body(std::istream& input,
                                     const int schema_version,
                                     const double electron_tolerance)
{
    FrozenDensityArtifact artifact;
    artifact.schema_version = schema_version;
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
    input >> artifact.grid_x >> artifact.grid_y >> artifact.grid_z
          >> artifact.cell_volume_bohr3;
    require_token(input, "POPULATIONS");
    input >> artifact.alpha_electrons >> artifact.beta_electrons;
    require_token(input, "RHO_UNIFORM");
    double alpha = 0.0;
    double beta = 0.0;
    input >> alpha >> beta;
    require_token(input, "END");
    if (!input)
    {
        throw std::invalid_argument("FDE uniform density seed is truncated");
    }
    require_no_trailing_content(input, "FDE uniform density seed");
    const std::size_t size = artifact_grid_size(artifact);
    artifact.freeze_thaw_cycle = 0;
    artifact.scf_converged = false;
    artifact.scf_iterations = 0;
    artifact.scf_density_residual = 0.0;
    artifact.orbital_kinetic_energy_ry = 0.0;
    artifact.nonlocal_pseudopotential_energy_ry = 0.0;
    artifact.rho_alpha_bohr3.assign(size, alpha);
    artifact.rho_beta_bohr3.assign(size, beta);
    DensityArtifactIO::validate(artifact, electron_tolerance);
    return artifact;
}

} // namespace

void DensityArtifactIO::validate(const FrozenDensityArtifact& artifact,
                                 const double electron_tolerance)
{
    if (artifact.schema_version < 1
        || artifact.schema_version > current_artifact_schema_version)
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
    if (artifact.schema_version >= 2
        && (artifact.scf_iterations <= 0
            || !std::isfinite(artifact.scf_density_residual)
            || artifact.scf_density_residual < 0.0))
    {
        throw std::invalid_argument(
            "FDE density artifact SCF iterations and residual are invalid");
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

void DensityArtifactIO::validate_compatible_set(
    const std::vector<FrozenDensityArtifact>& artifacts,
    const double electron_tolerance)
{
    if (artifacts.size() < 2)
    {
        throw std::invalid_argument("FDE compatible artifact set requires at least two fragments");
    }
    std::set<std::string> labels;
    DensityArtifactIO::validate(artifacts[0], electron_tolerance);
    labels.insert(artifacts[0].fragment_label);
    for (std::size_t index = 1; index < artifacts.size(); ++index)
    {
        DensityArtifactIO::validate_compatible_pair(artifacts[0],
                                                    artifacts[index],
                                                    electron_tolerance);
        if (!labels.insert(artifacts[index].fragment_label).second)
        {
            throw std::invalid_argument("FDE artifact set contains a duplicate fragment label");
        }
    }
}

void DensityArtifactIO::write(std::ostream& output, const FrozenDensityArtifact& artifact)
{
    DensityArtifactIO::validate(artifact, 1.0e-8);
    output << std::setprecision(17);
    output << "FDE_DENSITY_ARTIFACT " << artifact.schema_version << '\n';
    write_metadata(output, artifact);
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

void DensityArtifactIO::write_binary(std::ostream& output,
                                     const FrozenDensityArtifact& artifact)
{
    DensityArtifactIO::validate(artifact, 1.0e-8);
    output << std::setprecision(17);
    output << "FDE_DENSITY_BINARY " << current_binary_format_version << ' '
           << artifact.schema_version << '\n';
    write_metadata(output, artifact);
    output << "BYTE_ORDER LITTLE_ENDIAN\n";
    output << "RHO_ALPHA_BINARY " << artifact.rho_alpha_bohr3.size() << '\n';
    write_binary_density(output, artifact.rho_alpha_bohr3);
    output << '\n';
    output << "RHO_BETA_BINARY " << artifact.rho_beta_bohr3.size() << '\n';
    write_binary_density(output, artifact.rho_beta_bohr3);
    output << "\nEND\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE binary density artifact");
    }
}

FrozenDensityArtifact DensityArtifactIO::read(std::istream& input,
                                              const double electron_tolerance)
{
    require_token(input, "FDE_DENSITY_ARTIFACT");
    int schema_version = 0;
    input >> schema_version;
    return read_text_body(input, schema_version, electron_tolerance);
}

FrozenDensityArtifact DensityArtifactIO::read_binary(
    std::istream& input,
    const double electron_tolerance)
{
    require_token(input, "FDE_DENSITY_BINARY");
    int format_version = 0;
    int schema_version = 0;
    input >> format_version >> schema_version;
    return read_binary_body(input,
                            format_version,
                            schema_version,
                            electron_tolerance);
}

FrozenDensityArtifact DensityArtifactIO::read_runtime(std::istream& input,
                                                      const double electron_tolerance)
{
    std::string kind;
    input >> kind;
    if (kind == "FDE_DENSITY_ARTIFACT")
    {
        int schema_version = 0;
        input >> schema_version;
        return read_text_body(input, schema_version, electron_tolerance);
    }
    if (kind == "FDE_DENSITY_BINARY")
    {
        int format_version = 0;
        int schema_version = 0;
        input >> format_version >> schema_version;
        return read_binary_body(input,
                                format_version,
                                schema_version,
                                electron_tolerance);
    }
    if (kind != "FDE_UNIFORM_DENSITY_SEED")
    {
        throw std::invalid_argument("Unsupported FDE runtime density artifact header");
    }

    int schema_version = 0;
    input >> schema_version;
    return read_seed_body(input, schema_version, electron_tolerance);
}

} // namespace fde
