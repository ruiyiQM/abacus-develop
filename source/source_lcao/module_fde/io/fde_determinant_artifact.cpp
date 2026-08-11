#include "source_lcao/module_fde/io/fde_determinant_artifact.h"

#include "source_lcao/module_fde/embedding/fde_ao_projection.h"

#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>

namespace fde
{

namespace
{

const int artifact_schema_version = 1;

void validate_label(const std::string& value, const char* description)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(std::string("FDE determinant ") + description
                                    + " must be nonempty and contain no whitespace");
    }
}

void validate_overlap(const std::vector<double>& overlap,
                      const std::size_t dimension,
                      const double tolerance)
{
    if (dimension == 0 || overlap.size() != dimension * dimension
        || !std::isfinite(tolerance) || tolerance < 0.0)
    {
        throw std::invalid_argument("FDE determinant AO overlap contract is invalid");
    }
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            const double value = overlap[row + column * dimension];
            if (!std::isfinite(value)
                || std::fabs(value - overlap[column + row * dimension]) > tolerance)
            {
                throw std::invalid_argument("FDE determinant AO overlap must be finite and symmetric");
            }
        }
    }
}

double metric_overlap(const OccupiedSpinOrbitals& orbitals,
                      const std::size_t dimension,
                      const std::vector<double>& overlap,
                      const std::size_t first,
                      const std::size_t second)
{
    double result = 0.0;
    for (std::size_t column = 0; column < dimension; ++column)
    {
        double overlap_times_second = 0.0;
        for (std::size_t row = 0; row < dimension; ++row)
        {
            overlap_times_second
                += overlap[column + row * dimension]
                   * orbitals.coefficients[row + second * dimension];
        }
        result += orbitals.coefficients[column + first * dimension]
                  * overlap_times_second;
    }
    return result;
}

void validate_spin_orbitals(const OccupiedSpinOrbitals& orbitals,
                            const std::size_t dimension,
                            const std::vector<double>& overlap,
                            const double tolerance)
{
    const std::size_t count
        = DeterminantArtifactIO::occupied_count(orbitals, dimension);
    if (count > dimension || orbitals.orbital_energies_ry.size() != count
        || orbitals.source_fragment_labels.size() != count)
    {
        throw std::invalid_argument("FDE determinant occupied-orbital metadata is inconsistent");
    }
    for (std::size_t orbital = 0; orbital < count; ++orbital)
    {
        if (!std::isfinite(orbitals.orbital_energies_ry[orbital]))
        {
            throw std::invalid_argument("FDE determinant orbital energy is non-finite");
        }
        validate_label(orbitals.source_fragment_labels[orbital], "source fragment label");
        for (std::size_t ao = 0; ao < dimension; ++ao)
        {
            if (!std::isfinite(orbitals.coefficients[ao + orbital * dimension]))
            {
                throw std::invalid_argument("FDE determinant coefficient is non-finite");
            }
        }
    }

    for (std::size_t first = 0; first < count; ++first)
    {
        for (std::size_t second = 0; second < count; ++second)
        {
            if (orbitals.source_fragment_labels[first]
                != orbitals.source_fragment_labels[second])
            {
                continue;
            }
            const double expected = first == second ? 1.0 : 0.0;
            if (std::fabs(metric_overlap(orbitals,
                                        dimension,
                                        overlap,
                                        first,
                                        second)
                          - expected)
                > tolerance)
            {
                throw std::invalid_argument(
                    "FDE determinant orbitals are not orthonormal within a fragment");
            }
        }
    }
}

void append_occupied(const std::string& fragment_label,
                     const std::vector<std::size_t>& active_orbitals,
                     const int expected_electrons,
                     const ActiveSubspaceSolution& solution,
                     const std::size_t full_dimension,
                     OccupiedSpinOrbitals& result)
{
    const std::size_t active_dimension = active_orbitals.size();
    if (expected_electrons < 0
        || solution.eigenvectors.size() != active_dimension * active_dimension
        || solution.eigenvalues_ry.size() != active_dimension
        || solution.occupations.size() != active_dimension)
    {
        throw std::invalid_argument("FDE determinant fragment solution dimensions are inconsistent");
    }
    int occupied = 0;
    for (std::size_t state = 0; state < active_dimension; ++state)
    {
        const double occupation = solution.occupations[state];
        if (!std::isfinite(occupation)
            || (std::fabs(occupation) > 1.0e-12
                && std::fabs(occupation - 1.0) > 1.0e-12))
        {
            throw std::invalid_argument(
                "FDE determinant requires integer zero-or-one spin occupations");
        }
        if (occupation < 0.5)
        {
            continue;
        }
        ++occupied;
        const std::size_t old_size = result.coefficients.size();
        result.coefficients.resize(old_size + full_dimension, 0.0);
        for (std::size_t active = 0; active < active_dimension; ++active)
        {
            result.coefficients[old_size + active_orbitals[active]]
                = solution.eigenvectors[active + state * active_dimension];
        }
        result.orbital_energies_ry.push_back(solution.eigenvalues_ry[state]);
        result.source_fragment_labels.push_back(fragment_label);
    }
    if (occupied != expected_electrons)
    {
        throw std::invalid_argument(
            "FDE determinant occupied orbitals do not reproduce a fragment spin population");
    }
}

void require_token(std::istream& input, const char* expected)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        throw std::invalid_argument(std::string("FDE determinant artifact expected token ")
                                    + expected);
    }
}

void write_spin(std::ostream& output,
                const char* label,
                const OccupiedSpinOrbitals& orbitals,
                const std::size_t dimension)
{
    const std::size_t count = DeterminantArtifactIO::occupied_count(orbitals, dimension);
    output << label << ' ' << count << '\n';
    for (std::size_t orbital = 0; orbital < count; ++orbital)
    {
        output << "ORBITAL " << orbitals.source_fragment_labels[orbital] << ' '
               << orbitals.orbital_energies_ry[orbital];
        for (std::size_t ao = 0; ao < dimension; ++ao)
        {
            output << ' ' << orbitals.coefficients[ao + orbital * dimension];
        }
        output << '\n';
    }
}

OccupiedSpinOrbitals read_spin(std::istream& input,
                               const char* label,
                               const std::size_t dimension)
{
    require_token(input, label);
    std::size_t count = 0;
    input >> count;
    OccupiedSpinOrbitals orbitals;
    orbitals.coefficients.resize(dimension * count);
    orbitals.orbital_energies_ry.resize(count);
    orbitals.source_fragment_labels.resize(count);
    for (std::size_t orbital = 0; orbital < count; ++orbital)
    {
        require_token(input, "ORBITAL");
        input >> orbitals.source_fragment_labels[orbital]
              >> orbitals.orbital_energies_ry[orbital];
        for (std::size_t ao = 0; ao < dimension; ++ao)
        {
            input >> orbitals.coefficients[ao + orbital * dimension];
        }
    }
    return orbitals;
}

} // namespace

std::size_t DeterminantArtifactIO::occupied_count(const OccupiedSpinOrbitals& orbitals,
                                                  const std::size_t ao_dimension)
{
    if (ao_dimension == 0 || orbitals.coefficients.size() % ao_dimension != 0)
    {
        throw std::invalid_argument("FDE determinant coefficient dimensions are inconsistent");
    }
    return orbitals.coefficients.size() / ao_dimension;
}

void DeterminantArtifactIO::validate(const DiabaticDeterminantArtifact& artifact,
                                     const std::vector<double>& full_overlap,
                                     const double orthonormality_tolerance)
{
    if (artifact.schema_version != artifact_schema_version)
    {
        throw std::invalid_argument("Unsupported FDE determinant artifact version");
    }
    validate_label(artifact.state_label, "state label");
    validate_label(artifact.geometry_fingerprint, "geometry fingerprint");
    validate_label(artifact.orbital_fingerprint, "orbital fingerprint");
    validate_overlap(full_overlap, artifact.ao_dimension, orthonormality_tolerance);
    validate_spin_orbitals(artifact.alpha,
                           artifact.ao_dimension,
                           full_overlap,
                           orthonormality_tolerance);
    validate_spin_orbitals(artifact.beta,
                           artifact.ao_dimension,
                           full_overlap,
                           orthonormality_tolerance);
}

DiabaticDeterminantArtifact DeterminantArtifactIO::build(
    const DeterminantArtifactMetadata& metadata,
    const std::size_t full_ao_dimension,
    const std::vector<FragmentOrbitalSolution>& fragments,
    const std::vector<double>& full_overlap,
    const double orthonormality_tolerance)
{
    if (fragments.size() < 2)
    {
        throw std::invalid_argument("FDE determinant requires at least two fragment solutions");
    }
    std::set<std::string> labels;
    std::set<std::size_t> assigned_orbitals;
    DiabaticDeterminantArtifact artifact;
    artifact.schema_version = artifact_schema_version;
    artifact.state_label = metadata.state_label;
    artifact.geometry_fingerprint = metadata.geometry_fingerprint;
    artifact.orbital_fingerprint = metadata.orbital_fingerprint;
    artifact.ao_dimension = full_ao_dimension;
    for (std::size_t fragment_index = 0; fragment_index < fragments.size(); ++fragment_index)
    {
        const FragmentOrbitalSolution& fragment = fragments[fragment_index];
        validate_label(fragment.fragment_label, "fragment label");
        if (!labels.insert(fragment.fragment_label).second)
        {
            throw std::invalid_argument("FDE determinant fragment labels must be unique");
        }
        ActiveAoProjection::validate(full_ao_dimension, fragment.active_orbitals);
        for (std::size_t index = 0; index < fragment.active_orbitals.size(); ++index)
        {
            if (!assigned_orbitals.insert(fragment.active_orbitals[index]).second)
            {
                throw std::invalid_argument("FDE determinant fragment AO spaces must not overlap");
            }
        }
        append_occupied(fragment.fragment_label,
                        fragment.active_orbitals,
                        fragment.alpha_electrons,
                        fragment.alpha_solution,
                        full_ao_dimension,
                        artifact.alpha);
        append_occupied(fragment.fragment_label,
                        fragment.active_orbitals,
                        fragment.beta_electrons,
                        fragment.beta_solution,
                        full_ao_dimension,
                        artifact.beta);
    }
    DeterminantArtifactIO::validate(artifact,
                                    full_overlap,
                                    orthonormality_tolerance);
    return artifact;
}

void DeterminantArtifactIO::write(std::ostream& output,
                                  const DiabaticDeterminantArtifact& artifact,
                                  const std::vector<double>& full_overlap,
                                  const double orthonormality_tolerance)
{
    DeterminantArtifactIO::validate(artifact, full_overlap, orthonormality_tolerance);
    output << std::setprecision(17);
    output << "FDE_DIABATIC_DETERMINANT " << artifact.schema_version << '\n';
    output << "STATE " << artifact.state_label << '\n';
    output << "GEOMETRY " << artifact.geometry_fingerprint << '\n';
    output << "ORBITALS " << artifact.orbital_fingerprint << '\n';
    output << "AO_DIMENSION " << artifact.ao_dimension << '\n';
    write_spin(output, "ALPHA", artifact.alpha, artifact.ao_dimension);
    write_spin(output, "BETA", artifact.beta, artifact.ao_dimension);
    output << "END\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE determinant artifact");
    }
}

DiabaticDeterminantArtifact DeterminantArtifactIO::read(
    std::istream& input,
    const std::vector<double>& full_overlap,
    const double orthonormality_tolerance)
{
    DiabaticDeterminantArtifact artifact;
    require_token(input, "FDE_DIABATIC_DETERMINANT");
    input >> artifact.schema_version;
    require_token(input, "STATE");
    input >> artifact.state_label;
    require_token(input, "GEOMETRY");
    input >> artifact.geometry_fingerprint;
    require_token(input, "ORBITALS");
    input >> artifact.orbital_fingerprint;
    require_token(input, "AO_DIMENSION");
    input >> artifact.ao_dimension;
    artifact.alpha = read_spin(input, "ALPHA", artifact.ao_dimension);
    artifact.beta = read_spin(input, "BETA", artifact.ao_dimension);
    require_token(input, "END");
    if (!input)
    {
        throw std::invalid_argument("FDE determinant artifact is truncated or malformed");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE determinant artifact has trailing content");
    }
    DeterminantArtifactIO::validate(artifact, full_overlap, orthonormality_tolerance);
    return artifact;
}

} // namespace fde
