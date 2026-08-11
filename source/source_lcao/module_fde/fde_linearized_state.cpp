#include "fde_linearized_state.h"

#include <algorithm>
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

void require_token(std::istream& input, const char* expected)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        throw std::invalid_argument(std::string("FDE linearized state expected token ")
                                    + expected);
    }
}

void validate_matrix(const std::vector<double>& matrix,
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
            if (!std::isfinite(matrix[row + column * dimension])
                || std::fabs(matrix[row + column * dimension]
                             - matrix[column + row * dimension]) > tolerance)
            {
                throw std::invalid_argument("FDE linearized Hamiltonian must be finite and symmetric");
            }
        }
    }
}

void write_matrix(std::ostream& output,
                  const char* label,
                  const std::vector<double>& matrix)
{
    output << label << ' ' << matrix.size();
    for (std::size_t index = 0; index < matrix.size(); ++index)
    {
        output << ' ' << matrix[index];
    }
    output << '\n';
}

std::vector<double> read_matrix(std::istream& input,
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
    std::vector<double> result(size, 0.0);
    for (std::size_t index = 0; index < size; ++index)
    {
        input >> result[index];
    }
    return result;
}

double contraction(const std::vector<double>& density,
                   const std::vector<double>& hamiltonian,
                   const std::size_t dimension)
{
    if (density.size() != dimension * dimension
        || hamiltonian.size() != dimension * dimension)
    {
        throw std::invalid_argument("FDE transition-energy contraction dimensions are inconsistent");
    }
    double result = 0.0;
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

void LinearizedStateArtifactIO::validate(const LinearizedStateArtifact& artifact,
                                         const double symmetry_tolerance)
{
    if (artifact.schema_version != 1 || artifact.ao_dimension == 0
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

void LinearizedStateArtifactIO::write(std::ostream& output,
                                      const LinearizedStateArtifact& artifact,
                                      const double symmetry_tolerance)
{
    LinearizedStateArtifactIO::validate(artifact, symmetry_tolerance);
    output << std::setprecision(17);
    output << "FDE_LINEARIZED_STATE 1\n";
    output << "STATE " << artifact.state_label << '\n';
    output << "GEOMETRY " << artifact.geometry_fingerprint << '\n';
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

LinearizedStateArtifact LinearizedStateArtifactIO::read(
    std::istream& input,
    const double symmetry_tolerance)
{
    LinearizedStateArtifact result;
    require_token(input, "FDE_LINEARIZED_STATE");
    input >> result.schema_version;
    require_token(input, "STATE");
    input >> result.state_label;
    require_token(input, "GEOMETRY");
    input >> result.geometry_fingerprint;
    require_token(input, "AO_DIMENSION");
    input >> result.ao_dimension;
    require_token(input, "REFERENCE_ENERGY_RY");
    input >> result.reference_energy_ry;
    const std::size_t matrix_size = result.ao_dimension * result.ao_dimension;
    result.hamiltonian_alpha_ry
        = read_matrix(input, "HAMILTONIAN_ALPHA_RY", matrix_size);
    result.hamiltonian_beta_ry
        = read_matrix(input, "HAMILTONIAN_BETA_RY", matrix_size);
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
    LinearizedStateArtifactIO::validate(result, symmetry_tolerance);
    return result;
}

LinearizedTransitionEnergy::LinearizedTransitionEnergy(
    const std::vector<LinearizedStateArtifact>& states,
    const std::vector<double>& ao_overlap,
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
        LinearizedStateArtifactIO::validate(states_[index], 1.0e-8);
        if (states_[index].ao_dimension != dimension
            || states_[index].geometry_fingerprint != states_[0].geometry_fingerprint
            || !labels.insert(states_[index].state_label).second)
        {
            throw std::invalid_argument("FDE linearized states are incompatible or duplicated");
        }
    }
}

std::string LinearizedTransitionEnergy::name() const
{
    return "symmetric_linearized";
}

double LinearizedTransitionEnergy::evaluate_ry(
    const DiabaticDeterminantArtifact& bra,
    const DiabaticDeterminantArtifact& ket,
    const SpinTransitionDensityMatrix& transition_density,
    const std::size_t ao_dimension) const
{
    (void)ket;
    const LinearizedStateArtifact* state = nullptr;
    for (std::size_t index = 0; index < states_.size(); ++index)
    {
        if (states_[index].state_label == bra.state_label)
        {
            state = &states_[index];
            break;
        }
    }
    if (state == nullptr || state->ao_dimension != ao_dimension
        || state->geometry_fingerprint != bra.geometry_fingerprint)
    {
        throw std::invalid_argument("FDE transition determinant has no compatible linearized state");
    }
    if (transition_density.alpha.size() != ao_dimension * ao_dimension
        || transition_density.beta.size() != ao_dimension * ao_dimension)
    {
        throw std::invalid_argument("FDE transition density dimensions are inconsistent");
    }
    const DeterminantTransition self = ElectronicCoupling::transition(
        bra,
        bra,
        ao_overlap_,
        singular_value_tolerance_);
    std::vector<double> delta_alpha(ao_dimension * ao_dimension, 0.0);
    std::vector<double> delta_beta(ao_dimension * ao_dimension, 0.0);
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

} // namespace fde
