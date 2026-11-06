#include "source_lcao/module_fde/coupling/fde_diabatic_postprocess.h"

#include "source_lcao/module_fde/coupling/fde_coupling_provider_factory.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fde
{

namespace
{

std::ifstream open_input(const std::string& path, const char* description, std::ostream& running_log)
{
    std::ifstream input(path.c_str());
    if (!input)
    {
        throw std::runtime_error(std::string("Cannot open FDE ") + description + ": " + path);
    }
    report_artifact_file(path, running_log, true);
    return input;
}

const std::string& artifact_path(const std::vector<RuntimeArtifactPath>& artifacts,
                                 const std::string& label,
                                 const char* description)
{
    const std::string* result = nullptr;
    for (std::size_t index = 0; index < artifacts.size(); ++index)
    {
        if (artifacts[index].label == label)
        {
            if (result != nullptr)
            {
                throw std::invalid_argument(std::string("Duplicate FDE ") + description
                                            + " for " + label);
            }
            result = &artifacts[index].path;
        }
    }
    if (result == nullptr)
    {
        throw std::invalid_argument(std::string("Missing FDE ") + description
                                    + " for " + label);
    }
    return *result;
}

double diagonal_energy(const FdeRuntimeConfig& config, const std::string& label)
{
    const double* result = nullptr;
    for (std::size_t index = 0; index < config.diagonal_energies.size(); ++index)
    {
        if (config.diagonal_energies[index].state_label == label)
        {
            if (result != nullptr)
            {
                throw std::invalid_argument("Duplicate FDE diagonal energy for " + label);
            }
            result = &config.diagonal_energies[index].energy_ry;
        }
    }
    if (result == nullptr)
    {
        throw std::invalid_argument("Missing FDE diagonal energy for " + label);
    }
    return *result;
}

void require_token(std::istream& input, const char* expected)
{
    std::string token;
    if (!(input >> token) || token != expected)
    {
        throw std::invalid_argument(std::string("FDE AO-overlap artifact expected token ") + expected);
    }
}

template <typename Scalar>
struct AoOverlapInput
{
    std::vector<Scalar> values;
    std::vector<double> boundary_twist;
    std::string geometry_fingerprint;
    std::string orbital_fingerprint;
    std::string resource_fingerprint;
};

template <typename Scalar>
AoOverlapInput<Scalar> read_ao_overlap(const std::string& path, std::ostream& running_log)
{
    std::ifstream input = open_input(path, "AO-overlap artifact", running_log);
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    const int expected_version = complex_scalar ? 2 : 1;
    std::string token;
    int version = 0;
    if (!(input >> token >> version) || token != "FDE_AO_MATRIX" || version != expected_version)
    {
        throw std::invalid_argument("FDE AO-overlap scalar type/version mismatch");
    }
    AoOverlapInput<Scalar> result;
    if (complex_scalar)
    {
        require_token(input, "GEOMETRY");
        input >> result.geometry_fingerprint;
        require_token(input, "ORBITALS");
        input >> result.orbital_fingerprint;
        require_token(input, "BOUNDARY_TWIST");
        result.boundary_twist.resize(3);
        input >> result.boundary_twist[0] >> result.boundary_twist[1] >> result.boundary_twist[2];
        require_token(input, "RESOURCES");
        input >> result.resource_fingerprint;
    }
    std::size_t dimension = 0;
    if (!(input >> token >> dimension) || token != "DIMENSION" || dimension == 0)
    {
        throw std::invalid_argument("Invalid FDE AO-overlap dimension");
    }
    const std::size_t maximum_size = std::numeric_limits<std::size_t>::max();
    if (dimension > maximum_size / dimension)
    {
        throw std::invalid_argument("FDE AO-overlap matrix dimensions overflow");
    }
    std::size_t size = 0;
    if (!(input >> token >> size) || token != "VALUES" || size != dimension * dimension)
    {
        throw std::invalid_argument("Invalid FDE AO-overlap matrix size");
    }
    result.values.resize(size);
    for (std::size_t index = 0; index < size; ++index)
    {
        if (!(input >> result.values[index]))
        {
            throw std::invalid_argument("FDE AO-overlap matrix contains malformed values");
        }
        const double magnitude = std::abs(result.values[index]);
        if (!std::isfinite(magnitude))
        {
            throw std::invalid_argument("FDE AO-overlap matrix is non-finite");
        }
    }
    require_token(input, "END");
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE AO-overlap artifact has trailing content");
    }
    return result;
}

std::vector<std::string> all_fragment_labels(const FdeRuntimeConfig& config)
{
    std::vector<std::string> labels;
    for (std::size_t index = 0; index < config.fragments.size(); ++index)
    {
        labels.push_back(config.fragments[index].label);
    }
    return labels;
}

template <typename Scalar>
std::vector<double> validate_state_inputs(
    const FdeRuntimeConfig& config,
    const std::vector<ScalarDiabaticDeterminantArtifact<Scalar>>& determinants,
    const std::vector<ScalarLinearizedStateArtifact<Scalar>>& linearized_states)
{
    FdeRuntimeConfigIO::validate(config);
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    const bool finite_boundary = !config.boundary_twist.empty();
    if (complex_scalar != finite_boundary)
    {
        throw std::invalid_argument("FDE postprocess scalar artifacts do not match the declared boundary contract");
    }
    if (determinants.size() != config.states.size()
        || linearized_states.size() != config.states.size())
    {
        throw std::invalid_argument("FDE postprocess requires determinant and linearized data for every state");
    }
    std::vector<double> energies;
    for (std::size_t index = 0; index < config.states.size(); ++index)
    {
        const RuntimeStateDefinition& requested = config.states[index];
        const auto& determinant = determinants[index];
        const std::string& label = requested.state.label;
        if (determinant.state_label != label || linearized_states[index].state_label != label)
        {
            throw std::invalid_argument("FDE postprocess artifact ordering does not match state definitions");
        }
        if (determinant.boundary_twist != config.boundary_twist
            || linearized_states[index].boundary_twist != config.boundary_twist)
        {
            throw std::invalid_argument("FDE postprocess artifact boundary differs from the runtime state");
        }
        const double energy = diagonal_energy(config, label);
        const double difference = linearized_states[index].reference_energy_ry - energy;
        const double error = std::abs(difference);
        if (error > config.energy_tolerance_ry)
        {
            throw std::invalid_argument("FDE linearized reference and diagonal energy disagree");
        }
        std::size_t expected_alpha = 0;
        std::size_t expected_beta = 0;
        for (const auto& assignment : requested.state.fragments)
        {
            const std::size_t fragment_index
                = FdeRuntimeConfigIO::fragment_index(config, assignment.fragment_label);
            const FragmentDefinition& fragment = config.fragments[fragment_index];
            const SpinPopulation population = StateDefinition::spin_population(fragment, assignment);
            const auto& alpha_labels = determinant.alpha.source_fragment_labels;
            const auto& beta_labels = determinant.beta.source_fragment_labels;
            const std::size_t alpha = std::count(alpha_labels.begin(), alpha_labels.end(), fragment.label);
            const std::size_t beta = std::count(beta_labels.begin(), beta_labels.end(), fragment.label);
            if (alpha != static_cast<std::size_t>(population.alpha)
                || beta != static_cast<std::size_t>(population.beta))
            {
                throw std::invalid_argument("FDE determinant spin populations disagree with state "
                                            + label + " fragment " + fragment.label);
            }
            expected_alpha += static_cast<std::size_t>(population.alpha);
            expected_beta += static_cast<std::size_t>(population.beta);
        }
        const std::size_t alpha_count = ScalarDeterminantArtifactIO<Scalar>::occupied_count(
            determinant.alpha, determinant.ao_dimension);
        const std::size_t beta_count = ScalarDeterminantArtifactIO<Scalar>::occupied_count(
            determinant.beta, determinant.ao_dimension);
        if (alpha_count != expected_alpha || beta_count != expected_beta
            || determinant.alpha.source_fragment_labels.size() != alpha_count
            || determinant.beta.source_fragment_labels.size() != beta_count)
        {
            throw std::invalid_argument("FDE determinant explicit-electron sector disagrees with state " + label);
        }
        energies.push_back(energy);
    }
    return energies;
}

template <typename Scalar>
void write_result(const std::string& prefix,
                  const ScalarDiabaticPostprocessResult<Scalar>& result,
                  std::ostream& running_log)
{
    const std::string text_path = prefix + ".fde_diabatic";
    std::ofstream output(text_path.c_str());
    if (!output)
    {
        throw std::runtime_error("Cannot create FDE diabatic result: " + text_path);
    }
    output << std::setprecision(17);
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    const int version = complex_scalar ? 2 : 1;
    output << "FDE_DIABATIC_RESULT " << version << '\n';
    if (complex_scalar)
    {
        output << "GEOMETRY " << result.geometry_fingerprint << '\n';
        output << "ORBITALS " << result.orbital_fingerprint << '\n';
        output << "BOUNDARY_TWIST";
        for (const double twist : result.boundary_twist)
        {
            output << ' ' << twist;
        }
        output << "\nRESOURCES " << result.resource_fingerprint << '\n';
    }
    if (result.assembly.pairs.empty())
    {
        throw std::runtime_error("FDE diabatic result contains no state pairs");
    }
    output << "COUPLING_PROVIDER " << result.assembly.pairs[0].provider << '\n';
    output << "STATES " << result.assembly.problem.state_labels.size();
    for (std::size_t index = 0; index < result.assembly.problem.state_labels.size(); ++index)
    {
        output << ' ' << result.assembly.problem.state_labels[index];
    }
    output << '\n';
    output << "HAMILTONIAN_RY " << result.assembly.problem.hamiltonian_ry.size();
    for (std::size_t index = 0; index < result.assembly.problem.hamiltonian_ry.size(); ++index)
    {
        output << ' ' << result.assembly.problem.hamiltonian_ry[index];
    }
    output << '\n';
    output << "OVERLAP " << result.assembly.problem.overlap.size();
    for (std::size_t index = 0; index < result.assembly.problem.overlap.size(); ++index)
    {
        output << ' ' << result.assembly.problem.overlap[index];
    }
    output << '\n';
    for (std::size_t index = 0; index < result.assembly.pairs.size(); ++index)
    {
        const auto& pair = result.assembly.pairs[index];
        output << "PAIR " << pair.first_input_state << ' ' << pair.second_input_state << ' '
               << pair.normalized_overlap << ' ' << pair.hamiltonian_coupling_ry << ' '
               << result.orthogonalized_pair_couplings[index].coupling_ry << ' '
               << pair.overlap_reciprocity_error << ' '
               << pair.maximum_transition_density_trace_error << ' '
               << pair.transition_energy_asymmetry_ry << ' '
               << pair.estimated_coupling_uncertainty_ry << '\n';
    }
    output << "ADIABATIC_ENERGIES_RY " << result.adiabatic_solution.eigenvalues_ry.size();
    for (std::size_t index = 0; index < result.adiabatic_solution.eigenvalues_ry.size(); ++index)
    {
        output << ' ' << result.adiabatic_solution.eigenvalues_ry[index];
    }
    output << '\n';
    output << "COEFFICIENTS " << result.adiabatic_solution.coefficients.size();
    for (std::size_t index = 0; index < result.adiabatic_solution.coefficients.size(); ++index)
    {
        output << ' ' << result.adiabatic_solution.coefficients[index];
    }
    output << "\nMAXIMUM_RESIDUAL " << result.adiabatic_solution.maximum_residual
           << "\nEND\n";
    output.close();
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE diabatic result: " + text_path);
    }

    report_artifact_file(text_path, running_log, false);
    const std::string table_path = prefix + ".fde_diabatic.tsv";
    std::ofstream table(table_path.c_str());
    if (!table)
    {
        throw std::runtime_error("Cannot create FDE diabatic table: " + table_path);
    }
    table << std::setprecision(17);
    table << "kind\tfirst\tsecond\toverlap\th12_ry\torthogonalized_coupling_ry"
          << "\tprovider\toverlap_reciprocity_error"
          << "\tmaximum_transition_density_trace_error"
          << "\ttransition_energy_asymmetry_ry"
          << "\testimated_coupling_uncertainty_ry\n";
    for (std::size_t index = 0; index < result.assembly.pairs.size(); ++index)
    {
        const auto& pair = result.assembly.pairs[index];
        table << "pair\t" << pair.first_input_state
              << '\t' << pair.second_input_state
              << '\t' << pair.normalized_overlap << '\t' << pair.hamiltonian_coupling_ry
              << '\t' << result.orthogonalized_pair_couplings[index].coupling_ry
              << '\t' << pair.provider
              << '\t' << pair.overlap_reciprocity_error
              << '\t' << pair.maximum_transition_density_trace_error
              << '\t' << pair.transition_energy_asymmetry_ry
              << '\t' << pair.estimated_coupling_uncertainty_ry << '\n';
    }
    for (std::size_t root = 0; root < result.adiabatic_solution.eigenvalues_ry.size(); ++root)
    {
        table << "adiabatic_root\t" << root << "\t-\t-\t"
              << result.adiabatic_solution.eigenvalues_ry[root]
              << "\t-\t-\t-\t-\t-\t-\n";
    }
    table.close();
    if (!table)
    {
        throw std::runtime_error("Failed to write FDE diabatic table: " + table_path);
    }
    report_artifact_file(table_path, running_log, false);
}

void finish_solution(DiabaticPostprocessResult& result,
                      const FdeRuntimeConfig& config,
                      const std::vector<double>& diagonal_energies)
{
    NonorthogonalSolverControls solver_controls;
    solver_controls.overlap_eigenvalue_cutoff = config.overlap_eigenvalue_cutoff;
    solver_controls.symmetry_tolerance = config.symmetry_tolerance;
    solver_controls.residual_tolerance = config.residual_tolerance;
    result.adiabatic_solution
        = NonorthogonalMultistateSolver::solve(result.assembly.problem, solver_controls);
    for (std::size_t index = 0; index < result.assembly.pairs.size(); ++index)
    {
        const auto& pair = result.assembly.pairs[index];
        const double overlap = pair.normalized_overlap;
        const double denominator = 1.0 - overlap * overlap;
        if (denominator <= config.singular_value_tolerance)
        {
            throw std::runtime_error("FDE pair overlap is too large for symmetric orthogonalization");
        }
        const double first = diagonal_energies[pair.first_input_state];
        const double second = diagonal_energies[pair.second_input_state];
        const double coupling = (pair.hamiltonian_coupling_ry
                                 - 0.5 * overlap * (first + second))
                                / denominator;
        result.orthogonalized_pair_couplings.push_back(
            {pair.first_input_state, pair.second_input_state, coupling});
    }
}

void finish_solution(ComplexDiabaticPostprocessResult& result,
                      const FdeRuntimeConfig& config,
                      const std::vector<double>& diagonal_energies)
{
    if (config.states.size() != 2 || config.fragments.size() != 2
        || result.assembly.problem.state_labels.size() != 2 || result.assembly.pairs.size() != 1
        || !result.assembly.audit.omitted_state_indices.empty()
        || !result.assembly.audit.omitted_determinant_fragments.empty()
        || result.assembly.audit.discarded_interfragment_overlap_pairs != 0)
    {
        throw std::invalid_argument("FDE finite-complex local CT requires two complete states and both fragments");
    }
    const auto& pair = result.assembly.pairs[0];
    const double first = diagonal_energies[pair.first_input_state];
    const double second = diagonal_energies[pair.second_input_state];
    const auto solution = ComplexElectronicCoupling::orthogonalize_pair(
        first, second, pair.normalized_overlap, pair.hamiltonian_coupling_ry,
        config.overlap_eigenvalue_cutoff);
    if (solution.maximum_residual > config.residual_tolerance
        || solution.maximum_metric_error > config.symmetry_tolerance)
    {
        throw std::runtime_error("FDE finite-complex eigenvectors exceed residual or metric tolerance");
    }
    result.adiabatic_solution.diabatic_dimension = 2;
    result.adiabatic_solution.eigenvalues_ry = solution.eigenvalues_ry;
    result.adiabatic_solution.coefficients = solution.coefficients;
    const double overlap_magnitude = std::abs(pair.normalized_overlap);
    result.adiabatic_solution.retained_overlap_eigenvalues
        = {1.0 - overlap_magnitude, 1.0 + overlap_magnitude};
    result.adiabatic_solution.discarded_dimensions = 0;
    result.adiabatic_solution.maximum_residual = solution.maximum_residual;
    result.orthogonalized_pair_couplings.push_back(
        {pair.first_input_state, pair.second_input_state, solution.coupling_ry});
}

template <typename Scalar>
ScalarDiabaticPostprocessResult<Scalar> evaluate_impl(
    const FdeRuntimeConfig& config,
    const std::vector<ScalarDiabaticDeterminantArtifact<Scalar>>& determinants,
    const std::vector<ScalarLinearizedStateArtifact<Scalar>>& linearized_states,
    const std::vector<Scalar>& ao_overlap)
{
    const std::vector<double> diagonal_energies
        = validate_state_inputs(config, determinants, linearized_states);

    FdeDiabApproximationSpec approximation;
    const std::vector<std::string> selected_states
        = config.k_state_labels.empty()
              ? [&config]() {
                    std::vector<std::string> labels;
                    for (std::size_t index = 0; index < config.states.size(); ++index)
                    {
                        labels.push_back(config.states[index].state.label);
                    }
                    return labels;
                }()
              : config.k_state_labels;
    for (std::size_t selected = 0; selected < selected_states.size(); ++selected)
    {
        const std::size_t index = FdeRuntimeConfigIO::state_index(config, selected_states[selected]);
        approximation.state_indices.push_back(index);
    }
    approximation.overlap_active_fragments
        = config.l_fragment_labels.empty() ? all_fragment_labels(config)
                                           : config.l_fragment_labels;
    approximation.determinant_fragments
        = config.m_fragment_labels.empty() ? all_fragment_labels(config)
                                           : config.m_fragment_labels;

    const std::unique_ptr<ScalarTransitionEnergyProvider<Scalar>> transition_energy
        = ScalarFdeCouplingProviderFactory<Scalar>::create(config.coupling_provider,
                                             linearized_states,
                                             ao_overlap,
                                             config.singular_value_tolerance);
    CouplingValidationControls coupling_validation;
    coupling_validation.overlap_reciprocity_tolerance
        = config.symmetry_tolerance;
    coupling_validation.transition_density_trace_tolerance
        = config.transition_density_trace_tolerance;
    ScalarDiabaticPostprocessResult<Scalar> result;
    result.assembly = ScalarFdeDiabaticAssembler<Scalar>::assemble(determinants,
                                                     diagonal_energies,
                                                     ao_overlap,
                                                     approximation,
                                                     *transition_energy,
                                                     coupling_validation,
                                                     config.singular_value_tolerance);
    result.boundary_twist = config.boundary_twist;
    result.geometry_fingerprint = determinants[0].geometry_fingerprint;
    result.orbital_fingerprint = determinants[0].orbital_fingerprint;
    result.resource_fingerprint = determinants[0].resource_fingerprint;
    finish_solution(result, config, diagonal_energies);
    return result;
}

template <typename Scalar>
ScalarDiabaticPostprocessResult<Scalar> run_typed(const FdeRuntimeConfig& config,
                                                 std::ostream& running_log)
{
    if (config.ao_overlap_path.empty())
    {
        throw std::invalid_argument("FDE diabatic postprocess requires AO_OVERLAP");
    }
    const auto overlap = read_ao_overlap<Scalar>(config.ao_overlap_path, running_log);
    const bool complex_scalar = !std::is_same<Scalar, double>::value;
    std::vector<ScalarDiabaticDeterminantArtifact<Scalar>> determinants;
    std::vector<ScalarLinearizedStateArtifact<Scalar>> linearized_states;
    for (const auto& definition : config.states)
    {
        const std::string& label = definition.state.label;
        const std::string determinant_path = artifact_path(config.determinant_artifacts, label, "determinant artifact");
        std::ifstream determinant_input = open_input(determinant_path, "determinant artifact", running_log);
        const auto determinant = ScalarDeterminantArtifactIO<Scalar>::read(determinant_input, overlap.values, 1.0e-8);
        if (complex_scalar
            && (overlap.geometry_fingerprint != determinant.geometry_fingerprint
                || overlap.orbital_fingerprint != determinant.orbital_fingerprint
                || overlap.resource_fingerprint != determinant.resource_fingerprint
                || overlap.boundary_twist != determinant.boundary_twist))
        {
            throw std::invalid_argument("FDE AO-overlap identity does not match the finite determinant");
        }
        determinants.push_back(determinant);
        const std::string linear_path = artifact_path(config.linearized_state_artifacts, label, "linearized-state artifact");
        std::ifstream linear_input = open_input(linear_path, "linearized-state artifact", running_log);
        const auto linearized = ScalarLinearizedStateArtifactIO<Scalar>::read(linear_input, config.symmetry_tolerance);
        linearized_states.push_back(linearized);
    }
    const auto result = evaluate_impl(config, determinants, linearized_states, overlap.values);
    write_result(config.output_prefix, result, running_log);
    return result;
}

} // namespace

DiabaticPostprocessResult DiabaticPostprocessor::evaluate(
    const FdeRuntimeConfig& config,
    const std::vector<DiabaticDeterminantArtifact>& determinants,
    const std::vector<LinearizedStateArtifact>& linearized_states,
    const std::vector<double>& ao_overlap)
{
    return evaluate_impl(config, determinants, linearized_states, ao_overlap);
}

ComplexDiabaticPostprocessResult DiabaticPostprocessor::evaluate(
    const FdeRuntimeConfig& config,
    const std::vector<ComplexDiabaticDeterminantArtifact>& determinants,
    const std::vector<ComplexLinearizedStateArtifact>& linearized_states,
    const std::vector<std::complex<double>>& ao_overlap)
{
    return evaluate_impl(config, determinants, linearized_states, ao_overlap);
}

DiabaticPostprocessResult DiabaticPostprocessor::run(const std::string& config_path)
{
    std::ifstream input = open_input(config_path, "runtime config", std::clog);
    const FdeRuntimeConfig config = FdeRuntimeConfigIO::read(input);
    return run_typed<double>(config, std::clog);
}

std::size_t DiabaticPostprocessor::run_and_count_roots(const std::string& config_path,
                                                      std::ostream& running_log)
{
    std::ifstream input = open_input(config_path, "runtime config", running_log);
    const FdeRuntimeConfig config = FdeRuntimeConfigIO::read(input);
    if (config.boundary_twist.empty())
    {
        const auto result = run_typed<double>(config, running_log);
        return result.adiabatic_solution.eigenvalues_ry.size();
    }
    const auto result = run_typed<std::complex<double>>(config, running_log);
    return result.adiabatic_solution.eigenvalues_ry.size();
}

} // namespace fde
