#include "source_lcao/module_fde/coupling/fde_linearized_state.h"

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
        throw std::invalid_argument(std::string("FDE linearized state expected token ")
                                    + expected);
    }
}

template <typename Scalar>
void validate_matrix(const std::vector<Scalar>& matrix,
                     const std::size_t dimension,
                     const double tolerance)
{
    if (matrix.size() != dimension * dimension)
    {
        throw std::invalid_argument("FDE linearized Hamiltonian dimensions are inconsistent");
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
                throw std::invalid_argument("FDE linearized Hamiltonian must be finite and Hermitian");
            }
        }
    }
}

template <typename Scalar>
void write_matrix(std::ostream& output,
                  const char* label,
                  const std::vector<Scalar>& matrix)
{
    output << label << ' ' << matrix.size();
    for (std::size_t index = 0; index < matrix.size(); ++index)
    {
        output << ' ' << matrix[index];
    }
    output << '\n';
}

template <typename Scalar>
std::vector<Scalar> read_matrix(std::istream& input,
                                const char* label,
                                const std::size_t expected_size)
{
    require_token(input, label);
    std::size_t size = 0;
    input >> size;
    if (size != expected_size)
    {
        throw std::invalid_argument("FDE linearized Hamiltonian has an unexpected size");
    }
    std::vector<Scalar> result(size, 0.0);
    for (std::size_t index = 0; index < size; ++index)
    {
        input >> result[index];
    }
    return result;
}

template <typename Scalar>
Scalar contraction(const std::vector<Scalar>& density,
                   const std::vector<Scalar>& hamiltonian,
                   const std::size_t dimension)
{
    if (density.size() != dimension * dimension
        || hamiltonian.size() != dimension * dimension)
    {
        throw std::invalid_argument("FDE transition-energy contraction dimensions are inconsistent");
    }
    Scalar result = 0.0;
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            result += density[row + column * dimension]
                      * hamiltonian[column + row * dimension];
        }
    }
    return result;
}

} // namespace

template <typename Scalar>
void ScalarLinearizedStateArtifactIO<Scalar>::validate(const ScalarLinearizedStateArtifact<Scalar>& artifact,
                                         const double symmetry_tolerance)
{
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    validate_finite_state_boundary(artifact.schema_version, artifact.boundary_twist, complex_scalar);
    if (complex_scalar
        && (artifact.orbital_fingerprint.empty()
            || artifact.orbital_fingerprint.find_first_of(" \t\r\n") != std::string::npos))
    {
        throw std::invalid_argument("FDE complex linearized state requires an orbital fingerprint");
    }
    if (artifact.ao_dimension == 0
        || artifact.state_label.empty()
        || artifact.state_label.find_first_of(" \t\r\n") != std::string::npos
        || artifact.geometry_fingerprint.empty()
        || artifact.geometry_fingerprint.find_first_of(" \t\r\n") != std::string::npos
        || !std::isfinite(artifact.reference_energy_ry)
        || !std::isfinite(symmetry_tolerance) || symmetry_tolerance < 0.0)
    {
        throw std::invalid_argument("FDE linearized-state metadata or controls are invalid");
    }
    validate_matrix(artifact.hamiltonian_alpha_ry,
                    artifact.ao_dimension,
                    symmetry_tolerance);
    validate_matrix(artifact.hamiltonian_beta_ry,
                    artifact.ao_dimension,
                    symmetry_tolerance);
}

template <typename Scalar>
void ScalarLinearizedStateArtifactIO<Scalar>::write(std::ostream& output,
                                      const ScalarLinearizedStateArtifact<Scalar>& artifact,
                                      const double symmetry_tolerance)
{
    ScalarLinearizedStateArtifactIO<Scalar>::validate(artifact, symmetry_tolerance);
    output << std::setprecision(17);
    output << "FDE_LINEARIZED_STATE " << artifact.schema_version << '\n';
    output << "STATE " << artifact.state_label << '\n';
    output << "GEOMETRY " << artifact.geometry_fingerprint << '\n';
    if (!artifact.boundary_twist.empty())
    {
        output << "ORBITALS " << artifact.orbital_fingerprint << '\n';
        output << "BOUNDARY_TWIST";
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            output << ' ' << artifact.boundary_twist[axis];
        }
        output << '\n';
    }
    output << "AO_DIMENSION " << artifact.ao_dimension << '\n';
    output << "REFERENCE_ENERGY_RY " << artifact.reference_energy_ry << '\n';
    write_matrix(output, "HAMILTONIAN_ALPHA_RY", artifact.hamiltonian_alpha_ry);
    write_matrix(output, "HAMILTONIAN_BETA_RY", artifact.hamiltonian_beta_ry);
    output << "END\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE linearized-state artifact");
    }
}

template <typename Scalar>
ScalarLinearizedStateArtifact<Scalar> ScalarLinearizedStateArtifactIO<Scalar>::read(
    std::istream& input,
    const double symmetry_tolerance)
{
    ScalarLinearizedStateArtifact<Scalar> result;
    require_token(input, "FDE_LINEARIZED_STATE");
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    const int expected_version = complex_scalar ? 2 : 1;
    if (!(input >> result.schema_version) || result.schema_version != expected_version)
    {
        throw std::invalid_argument("FDE linearized-state scalar type/version mismatch");
    }
    require_token(input, "STATE");
    input >> result.state_label;
    require_token(input, "GEOMETRY");
    input >> result.geometry_fingerprint;
    if (complex_scalar)
    {
        require_token(input, "ORBITALS");
        input >> result.orbital_fingerprint;
        require_token(input, "BOUNDARY_TWIST");
        result.boundary_twist.resize(3);
        input >> result.boundary_twist[0] >> result.boundary_twist[1] >> result.boundary_twist[2];
    }
    require_token(input, "AO_DIMENSION");
    if (!(input >> result.ao_dimension) || result.ao_dimension == 0)
    {
        throw std::invalid_argument("FDE linearized-state AO dimension is malformed");
    }
    require_token(input, "REFERENCE_ENERGY_RY");
    input >> result.reference_energy_ry;
    const std::size_t matrix_size = result.ao_dimension * result.ao_dimension;
    result.hamiltonian_alpha_ry
        = read_matrix<Scalar>(input, "HAMILTONIAN_ALPHA_RY", matrix_size);
    result.hamiltonian_beta_ry
        = read_matrix<Scalar>(input, "HAMILTONIAN_BETA_RY", matrix_size);
    require_token(input, "END");
    if (!input)
    {
        throw std::invalid_argument("FDE linearized-state artifact is truncated");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE linearized-state artifact has trailing content");
    }
    ScalarLinearizedStateArtifactIO<Scalar>::validate(result, symmetry_tolerance);
    return result;
}

template <typename Scalar>
ScalarLinearizedTransitionEnergy<Scalar>::ScalarLinearizedTransitionEnergy(
    const std::vector<ScalarLinearizedStateArtifact<Scalar>>& states,
    const std::vector<Scalar>& ao_overlap,
    const double singular_value_tolerance)
    : states_(states),
      ao_overlap_(ao_overlap),
      singular_value_tolerance_(singular_value_tolerance)
{
    if (states_.size() < 2 || !std::isfinite(singular_value_tolerance_)
        || singular_value_tolerance_ <= 0.0)
    {
        throw std::invalid_argument("FDE linearized transition-energy inputs are invalid");
    }
    std::set<std::string> labels;
    const std::size_t dimension = states_[0].ao_dimension;
    if (ao_overlap_.size() != dimension * dimension)
    {
        throw std::invalid_argument("FDE linearized transition AO overlap has invalid dimensions");
    }
    for (std::size_t index = 0; index < states_.size(); ++index)
    {
        ScalarLinearizedStateArtifactIO<Scalar>::validate(states_[index], 1.0e-8);
        if (states_[index].ao_dimension != dimension
            || states_[index].geometry_fingerprint != states_[0].geometry_fingerprint
            || states_[index].orbital_fingerprint != states_[0].orbital_fingerprint
            || states_[index].boundary_twist != states_[0].boundary_twist
            || !labels.insert(states_[index].state_label).second)
        {
            throw std::invalid_argument("FDE linearized states are incompatible or duplicated");
        }
    }
}

template <typename Scalar>
std::string ScalarLinearizedTransitionEnergy<Scalar>::name() const
{
    return "symmetric_linearized";
}

template <typename Scalar>
Scalar ScalarLinearizedTransitionEnergy<Scalar>::evaluate_ry(
    const ScalarDiabaticDeterminantArtifact<Scalar>& bra,
    const ScalarDiabaticDeterminantArtifact<Scalar>& ket,
    const ScalarSpinTransitionDensityMatrix<Scalar>& transition_density,
    const std::size_t ao_dimension) const
{
    (void)ket;
    const ScalarLinearizedStateArtifact<Scalar>* state = nullptr;
    for (std::size_t index = 0; index < states_.size(); ++index)
    {
        if (states_[index].state_label == bra.state_label)
        {
            state = &states_[index];
            break;
        }
    }
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    if (state == nullptr || state->ao_dimension != ao_dimension
        || state->geometry_fingerprint != bra.geometry_fingerprint
        || state->boundary_twist != bra.boundary_twist
        || (complex_scalar && state->orbital_fingerprint != bra.orbital_fingerprint))
    {
        throw std::invalid_argument("FDE transition determinant has no compatible linearized state");
    }
    if (transition_density.alpha.size() != ao_dimension * ao_dimension
        || transition_density.beta.size() != ao_dimension * ao_dimension)
    {
        throw std::invalid_argument("FDE transition density dimensions are inconsistent");
    }
    const ScalarDeterminantTransition<Scalar> self = ScalarElectronicCoupling<Scalar>::transition(
        bra,
        bra,
        ao_overlap_,
        singular_value_tolerance_);
    std::vector<Scalar> delta_alpha(ao_dimension * ao_dimension, 0.0);
    std::vector<Scalar> delta_beta(ao_dimension * ao_dimension, 0.0);
    for (std::size_t index = 0; index < delta_alpha.size(); ++index)
    {
        delta_alpha[index]
            = transition_density.alpha[index] - self.density_matrix.alpha[index];
        delta_beta[index]
            = transition_density.beta[index] - self.density_matrix.beta[index];
    }
    return state->reference_energy_ry
           + contraction(delta_alpha, state->hamiltonian_alpha_ry, ao_dimension)
           + contraction(delta_beta, state->hamiltonian_beta_ry, ao_dimension);
}

template class ScalarLinearizedStateArtifactIO<double>;
template class ScalarLinearizedStateArtifactIO<std::complex<double>>;
template class ScalarLinearizedTransitionEnergy<double>;
template class ScalarLinearizedTransitionEnergy<std::complex<double>>;

} // namespace fde
