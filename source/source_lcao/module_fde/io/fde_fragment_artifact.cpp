#include "source_lcao/module_fde/io/fde_fragment_artifact.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fde
{

namespace
{

void require_token(std::istream& input, const char* expected)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        throw std::invalid_argument(std::string("FDE fragment artifact expected token ")
                                    + expected);
    }
}

void validate_token(const std::string& value, const char* description)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(std::string("FDE fragment artifact ") + description
                                    + " must be a nonempty token");
    }
}

template <typename Scalar>
void validate_matrix(const std::vector<Scalar>& matrix,
                     const std::size_t dimension,
                     const double tolerance,
                     const char* description)
{
    if (matrix.size() != dimension * dimension)
    {
        throw std::invalid_argument(std::string("FDE fragment ") + description
                                    + " dimensions are inconsistent");
    }
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            const Scalar value = matrix[row + column * dimension];
            const double magnitude = std::abs(value);
            const Scalar transpose = conjugate_scalar(matrix[column + row * dimension]);
            const Scalar difference = value - transpose;
            const double error = std::abs(difference);
            if (!std::isfinite(magnitude) || error > tolerance)
            {
                throw std::invalid_argument(std::string("FDE fragment ") + description
                                            + " must be finite and Hermitian");
            }
        }
    }
}

template <typename Scalar>
void validate_orbitals(const ScalarOccupiedSpinOrbitals<Scalar>& orbitals,
                       const std::size_t dimension,
                       const std::string& fragment_label,
                       const std::vector<std::size_t>& active_orbitals,
                       const double tolerance)
{
    const std::size_t count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(orbitals, dimension);
    if (count > active_orbitals.size()
        || orbitals.orbital_energies_ry.size() != count
        || orbitals.source_fragment_labels.size() != count)
    {
        throw std::invalid_argument("FDE fragment occupied-orbital metadata is inconsistent");
    }
    for (std::size_t orbital = 0; orbital < count; ++orbital)
    {
        if (!std::isfinite(orbitals.orbital_energies_ry[orbital])
            || orbitals.source_fragment_labels[orbital] != fragment_label)
        {
            throw std::invalid_argument("FDE fragment occupied orbital has invalid provenance");
        }
        for (std::size_t ao = 0; ao < dimension; ++ao)
        {
            const double magnitude = std::abs(orbitals.coefficients[ao + orbital * dimension]);
            if (!std::isfinite(magnitude))
            {
                throw std::invalid_argument("FDE fragment occupied coefficient is non-finite");
            }
            const bool owned = std::binary_search(active_orbitals.begin(), active_orbitals.end(), ao);
            if (!owned && magnitude > tolerance)
            {
                throw std::invalid_argument("FDE fragment occupied coefficient lies outside its owned AO space");
            }
        }
    }
}

template <typename Scalar>
void write_vector(std::ostream& output,
                  const char* label,
                  const std::vector<Scalar>& values)
{
    output << label << ' ' << values.size();
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        output << ' ' << values[index];
    }
    output << '\n';
}

template <typename Scalar>
std::vector<Scalar> read_vector(std::istream& input,
                                const char* label,
                                const std::size_t expected_size)
{
    require_token(input, label);
    std::size_t size = 0;
    input >> size;
    if (size != expected_size)
    {
        throw std::invalid_argument(std::string("FDE fragment ") + label
                                    + " has an unexpected size");
    }
    std::vector<Scalar> values(size, 0.0);
    for (std::size_t index = 0; index < size; ++index)
    {
        input >> values[index];
    }
    return values;
}

template <typename Scalar>
void write_spin(std::ostream& output,
                const char* label,
                const ScalarOccupiedSpinOrbitals<Scalar>& orbitals,
                const std::size_t dimension)
{
    const std::size_t count
        = ScalarDeterminantArtifactIO<Scalar>::occupied_count(orbitals, dimension);
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

template <typename Scalar>
ScalarOccupiedSpinOrbitals<Scalar> read_spin(std::istream& input,
                               const char* label,
                               const std::size_t dimension)
{
    require_token(input, label);
    std::size_t count = 0;
    if (!(input >> count) || count > dimension)
    {
        throw std::invalid_argument("FDE fragment occupied count exceeds AO dimension or is malformed");
    }
    ScalarOccupiedSpinOrbitals<Scalar> result;
    result.coefficients.assign(dimension * count, 0.0);
    result.orbital_energies_ry.assign(count, 0.0);
    result.source_fragment_labels.resize(count);
    for (std::size_t orbital = 0; orbital < count; ++orbital)
    {
        require_token(input, "ORBITAL");
        input >> result.source_fragment_labels[orbital]
              >> result.orbital_energies_ry[orbital];
        for (std::size_t ao = 0; ao < dimension; ++ao)
        {
            input >> result.coefficients[ao + orbital * dimension];
        }
    }
    return result;
}

} // namespace

template <typename Scalar>
void ScalarFragmentScfArtifactIO<Scalar>::validate(const ScalarFragmentScfArtifact<Scalar>& artifact,
                                     const double symmetry_tolerance)
{
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    validate_finite_state_boundary(artifact.schema_version, artifact.boundary_twist, complex_scalar);
    if (artifact.ao_dimension == 0
        || artifact.freeze_thaw_cycle < 0 || !artifact.scf_converged
        || !std::isfinite(symmetry_tolerance) || symmetry_tolerance < 0.0)
    {
        throw std::invalid_argument("FDE fragment artifact header or controls are invalid");
    }
    if (complex_scalar)
    {
        validate_token(artifact.resource_fingerprint, "resource fingerprint");
    }
    validate_token(artifact.state_label, "state label");
    validate_token(artifact.fragment_label, "fragment label");
    validate_token(artifact.geometry_fingerprint, "geometry fingerprint");
    validate_token(artifact.orbital_fingerprint, "orbital fingerprint");
    validate_token(artifact.density_path, "density path");

    if (artifact.active_orbitals.empty()
        || !std::is_sorted(artifact.active_orbitals.begin(), artifact.active_orbitals.end())
        || std::adjacent_find(artifact.active_orbitals.begin(), artifact.active_orbitals.end())
               != artifact.active_orbitals.end()
        || artifact.active_orbitals.back() >= artifact.ao_dimension)
    {
        throw std::invalid_argument("FDE fragment active AO list is invalid");
    }
    validate_orbitals(artifact.alpha,
                       artifact.ao_dimension,
                       artifact.fragment_label,
                       artifact.active_orbitals,
                       symmetry_tolerance);
    validate_orbitals(artifact.beta,
                       artifact.ao_dimension,
                       artifact.fragment_label,
                       artifact.active_orbitals,
                       symmetry_tolerance);
    validate_matrix(artifact.ao_overlap,
                    artifact.ao_dimension,
                    symmetry_tolerance,
                    "AO overlap");
    validate_matrix(artifact.hamiltonian_alpha_ry,
                    artifact.ao_dimension,
                    symmetry_tolerance,
                    "alpha Hamiltonian");
    validate_matrix(artifact.hamiltonian_beta_ry,
                    artifact.ao_dimension,
                    symmetry_tolerance,
                    "beta Hamiltonian");
    const double energies[] = {artifact.subsystem_total_energy_ry,
                               artifact.ion_ion_energy_ry,
                               artifact.hartree_cross_energy_ry,
                               artifact.nonadditive_kinetic_energy_ry,
                               artifact.nonadditive_xc_energy_ry};
    for (std::size_t index = 0; index < sizeof(energies) / sizeof(energies[0]); ++index)
    {
        if (!std::isfinite(energies[index]))
        {
            throw std::invalid_argument("FDE fragment artifact energy is non-finite");
        }
    }
}

template <typename Scalar>
void ScalarFragmentScfArtifactIO<Scalar>::write(std::ostream& output,
                                  const ScalarFragmentScfArtifact<Scalar>& artifact,
                                  const double symmetry_tolerance)
{
    ScalarFragmentScfArtifactIO<Scalar>::validate(artifact, symmetry_tolerance);
    output << std::setprecision(17);
    output << "FDE_FRAGMENT_SCF_ARTIFACT " << artifact.schema_version << '\n';
    output << "STATE " << artifact.state_label << '\n';
    output << "FRAGMENT " << artifact.fragment_label << '\n';
    output << "GEOMETRY " << artifact.geometry_fingerprint << '\n';
    output << "ORBITAL_BASIS " << artifact.orbital_fingerprint << '\n';
    if (!artifact.boundary_twist.empty())
    {
        output << "BOUNDARY_TWIST";
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            output << ' ' << artifact.boundary_twist[axis];
        }
        output << '\n';
        output << "RESOURCES " << artifact.resource_fingerprint << '\n';
    }
    output << "DENSITY_PATH " << artifact.density_path << '\n';
    output << "CYCLE " << artifact.freeze_thaw_cycle << '\n';
    output << "SCF_CONVERGED " << (artifact.scf_converged ? 1 : 0) << '\n';
    output << "AO_DIMENSION " << artifact.ao_dimension << '\n';
    output << "ACTIVE_ORBITALS " << artifact.active_orbitals.size();
    for (std::size_t index = 0; index < artifact.active_orbitals.size(); ++index)
    {
        output << ' ' << artifact.active_orbitals[index];
    }
    output << '\n';
    write_spin(output, "ALPHA", artifact.alpha, artifact.ao_dimension);
    write_spin(output, "BETA", artifact.beta, artifact.ao_dimension);
    write_vector(output, "AO_OVERLAP", artifact.ao_overlap);
    write_vector(output, "HAMILTONIAN_ALPHA_RY", artifact.hamiltonian_alpha_ry);
    write_vector(output, "HAMILTONIAN_BETA_RY", artifact.hamiltonian_beta_ry);
    output << "SUBSYSTEM_TOTAL_ENERGY_RY " << artifact.subsystem_total_energy_ry << '\n';
    output << "ION_ION_ENERGY_RY " << artifact.ion_ion_energy_ry << '\n';
    output << "HARTREE_CROSS_ENERGY_RY " << artifact.hartree_cross_energy_ry << '\n';
    output << "NONADDITIVE_KINETIC_ENERGY_RY "
           << artifact.nonadditive_kinetic_energy_ry << '\n';
    output << "NONADDITIVE_XC_ENERGY_RY " << artifact.nonadditive_xc_energy_ry << '\n';
    output << "END_FDE_FRAGMENT_SCF_ARTIFACT\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE fragment SCF artifact");
    }
}

template <typename Scalar>
ScalarFragmentScfArtifact<Scalar> ScalarFragmentScfArtifactIO<Scalar>::read(std::istream& input,
                                                const double symmetry_tolerance)
{
    require_token(input, "FDE_FRAGMENT_SCF_ARTIFACT");
    ScalarFragmentScfArtifact<Scalar> result;
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    const int expected_version = complex_scalar ? 2 : 1;
    if (!(input >> result.schema_version) || result.schema_version != expected_version)
    {
        throw std::invalid_argument("FDE fragment scalar type/version mismatch");
    }
    require_token(input, "STATE");
    input >> result.state_label;
    require_token(input, "FRAGMENT");
    input >> result.fragment_label;
    require_token(input, "GEOMETRY");
    input >> result.geometry_fingerprint;
    require_token(input, "ORBITAL_BASIS");
    input >> result.orbital_fingerprint;
    if (complex_scalar)
    {
        require_token(input, "BOUNDARY_TWIST");
        result.boundary_twist.resize(3);
        input >> result.boundary_twist[0] >> result.boundary_twist[1] >> result.boundary_twist[2];
        require_token(input, "RESOURCES");
        input >> result.resource_fingerprint;
    }
    require_token(input, "DENSITY_PATH");
    input >> result.density_path;
    require_token(input, "CYCLE");
    input >> result.freeze_thaw_cycle;
    require_token(input, "SCF_CONVERGED");
    int converged = 0;
    input >> converged;
    result.scf_converged = converged == 1;
    if (converged != 0 && converged != 1)
    {
        throw std::invalid_argument("FDE fragment SCF convergence flag must be zero or one");
    }
    require_token(input, "AO_DIMENSION");
    if (!(input >> result.ao_dimension) || result.ao_dimension == 0)
    {
        throw std::invalid_argument("FDE fragment AO dimension is malformed");
    }
    require_token(input, "ACTIVE_ORBITALS");
    std::size_t active_count = 0;
    if (!(input >> active_count) || active_count > result.ao_dimension)
    {
        throw std::invalid_argument("FDE fragment active count exceeds AO dimension or is malformed");
    }
    result.active_orbitals.resize(active_count);
    for (std::size_t index = 0; index < active_count; ++index)
    {
        input >> result.active_orbitals[index];
    }
    result.alpha = read_spin<Scalar>(input, "ALPHA", result.ao_dimension);
    result.beta = read_spin<Scalar>(input, "BETA", result.ao_dimension);
    const std::size_t matrix_size = result.ao_dimension * result.ao_dimension;
    result.ao_overlap = read_vector<Scalar>(input, "AO_OVERLAP", matrix_size);
    result.hamiltonian_alpha_ry
        = read_vector<Scalar>(input, "HAMILTONIAN_ALPHA_RY", matrix_size);
    result.hamiltonian_beta_ry
        = read_vector<Scalar>(input, "HAMILTONIAN_BETA_RY", matrix_size);
    require_token(input, "SUBSYSTEM_TOTAL_ENERGY_RY");
    input >> result.subsystem_total_energy_ry;
    require_token(input, "ION_ION_ENERGY_RY");
    input >> result.ion_ion_energy_ry;
    require_token(input, "HARTREE_CROSS_ENERGY_RY");
    input >> result.hartree_cross_energy_ry;
    require_token(input, "NONADDITIVE_KINETIC_ENERGY_RY");
    input >> result.nonadditive_kinetic_energy_ry;
    require_token(input, "NONADDITIVE_XC_ENERGY_RY");
    input >> result.nonadditive_xc_energy_ry;
    require_token(input, "END_FDE_FRAGMENT_SCF_ARTIFACT");
    if (!input)
    {
        throw std::invalid_argument("FDE fragment SCF artifact is truncated");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE fragment SCF artifact has trailing content");
    }
    ScalarFragmentScfArtifactIO<Scalar>::validate(result, symmetry_tolerance);
    return result;
}

template class ScalarFragmentScfArtifactIO<double>;
template class ScalarFragmentScfArtifactIO<std::complex<double>>;

} // namespace fde
