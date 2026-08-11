#include "source_lcao/module_fde/io/fde_kpoint_band_artifact.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

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
        throw std::invalid_argument(
            std::string("FDE k-point band artifact expected token ") + expected);
    }
}

void validate_label(const std::string& value, const char* description)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(
            std::string("FDE k-point band ") + description
            + " must be a nonempty token");
    }
}

bool close(const double first, const double second, const double tolerance)
{
    return std::fabs(first - second) <= tolerance;
}

} // namespace

void FdeKPointBandArtifactIO::validate(
    const FdeKPointBandArtifact& artifact,
    const double tolerance)
{
    if (artifact.schema_version != artifact_schema_version
        || artifact.ao_dimension == 0 || artifact.band_count == 0
        || artifact.band_count > artifact.ao_dimension
        || artifact.points.empty() || artifact.points.size() % 2 != 0
        || !std::isfinite(tolerance) || tolerance < 0.0)
    {
        throw std::invalid_argument("FDE k-point band artifact header is invalid");
    }
    validate_label(artifact.state_label, "state label");
    validate_label(artifact.fragment_label, "fragment label");
    validate_label(artifact.geometry_fingerprint, "geometry fingerprint");
    validate_label(artifact.orbital_fingerprint, "orbital fingerprint");

    std::vector<const FdeKPointBand*> by_spin[2];
    double weight_sum[2] = {0.0, 0.0};
    for (std::size_t index = 0; index < artifact.points.size(); ++index)
    {
        const FdeKPointBand& point = artifact.points[index];
        if (point.spin < 0 || point.spin > 1
            || point.eigenvalues_ry.size() != artifact.band_count
            || !std::isfinite(point.kx_direct)
            || !std::isfinite(point.ky_direct)
            || !std::isfinite(point.kz_direct)
            || !std::isfinite(point.weight) || point.weight < 0.0)
        {
            throw std::invalid_argument("FDE k-point band entry is invalid");
        }
        if (point.physical_kpoint
            != by_spin[point.spin].size())
        {
            throw std::invalid_argument(
                "FDE physical k-point indices must be contiguous within each spin");
        }
        for (std::size_t band = 0; band < point.eigenvalues_ry.size(); ++band)
        {
            if (!std::isfinite(point.eigenvalues_ry[band])
                || (band != 0
                    && point.eigenvalues_ry[band] + tolerance
                           < point.eigenvalues_ry[band - 1]))
            {
                throw std::invalid_argument(
                    "FDE k-point eigenvalues must be finite and ordered");
            }
        }
        by_spin[point.spin].push_back(&point);
        weight_sum[point.spin] += point.weight;
    }
    if (by_spin[0].empty() || by_spin[0].size() != by_spin[1].size()
        || !close(weight_sum[0], 1.0, tolerance)
        || !close(weight_sum[1], 1.0, tolerance))
    {
        throw std::invalid_argument(
            "FDE k-point band artifact requires paired normalized spin meshes");
    }
    for (std::size_t index = 0; index < by_spin[0].size(); ++index)
    {
        const FdeKPointBand& alpha = *by_spin[0][index];
        const FdeKPointBand& beta = *by_spin[1][index];
        if (!close(alpha.kx_direct, beta.kx_direct, tolerance)
            || !close(alpha.ky_direct, beta.ky_direct, tolerance)
            || !close(alpha.kz_direct, beta.kz_direct, tolerance)
            || !close(alpha.weight, beta.weight, tolerance))
        {
            throw std::invalid_argument(
                "FDE alpha and beta k-point meshes do not match");
        }
    }
}

void FdeKPointBandArtifactIO::write(
    std::ostream& output,
    const FdeKPointBandArtifact& artifact,
    const double tolerance)
{
    FdeKPointBandArtifactIO::validate(artifact, tolerance);
    output << std::setprecision(17);
    output << "FDE_KPOINT_BANDS " << artifact.schema_version << '\n';
    output << "STATE " << artifact.state_label << '\n';
    output << "FRAGMENT " << artifact.fragment_label << '\n';
    output << "GEOMETRY " << artifact.geometry_fingerprint << '\n';
    output << "ORBITALS " << artifact.orbital_fingerprint << '\n';
    output << "AO_DIMENSION " << artifact.ao_dimension << '\n';
    output << "BAND_COUNT " << artifact.band_count << '\n';
    output << "KPOINT_COUNT " << artifact.points.size() << '\n';
    for (std::size_t index = 0; index < artifact.points.size(); ++index)
    {
        const FdeKPointBand& point = artifact.points[index];
        output << "KPOINT " << point.spin << ' ' << point.physical_kpoint
               << ' ' << point.kx_direct << ' ' << point.ky_direct
               << ' ' << point.kz_direct << ' ' << point.weight << '\n';
        output << "EIGENVALUES " << point.eigenvalues_ry.size();
        for (std::size_t band = 0; band < point.eigenvalues_ry.size(); ++band)
        {
            output << ' ' << point.eigenvalues_ry[band];
        }
        output << '\n';
    }
    output << "END_FDE_KPOINT_BANDS\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE k-point band artifact");
    }
}

FdeKPointBandArtifact FdeKPointBandArtifactIO::read(
    std::istream& input,
    const double tolerance)
{
    FdeKPointBandArtifact artifact;
    require_token(input, "FDE_KPOINT_BANDS");
    input >> artifact.schema_version;
    require_token(input, "STATE");
    input >> artifact.state_label;
    require_token(input, "FRAGMENT");
    input >> artifact.fragment_label;
    require_token(input, "GEOMETRY");
    input >> artifact.geometry_fingerprint;
    require_token(input, "ORBITALS");
    input >> artifact.orbital_fingerprint;
    require_token(input, "AO_DIMENSION");
    input >> artifact.ao_dimension;
    require_token(input, "BAND_COUNT");
    input >> artifact.band_count;
    require_token(input, "KPOINT_COUNT");
    std::size_t point_count = 0;
    input >> point_count;
    artifact.points.resize(point_count);
    for (std::size_t index = 0; index < point_count; ++index)
    {
        FdeKPointBand& point = artifact.points[index];
        require_token(input, "KPOINT");
        input >> point.spin >> point.physical_kpoint
              >> point.kx_direct >> point.ky_direct >> point.kz_direct
              >> point.weight;
        require_token(input, "EIGENVALUES");
        std::size_t band_count = 0;
        input >> band_count;
        point.eigenvalues_ry.resize(band_count);
        for (std::size_t band = 0; band < band_count; ++band)
        {
            input >> point.eigenvalues_ry[band];
        }
    }
    require_token(input, "END_FDE_KPOINT_BANDS");
    if (!input)
    {
        throw std::invalid_argument(
            "FDE k-point band artifact is truncated or malformed");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument(
            "FDE k-point band artifact has trailing content");
    }
    FdeKPointBandArtifactIO::validate(artifact, tolerance);
    return artifact;
}

} // namespace fde
