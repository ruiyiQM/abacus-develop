#include "source_lcao/module_fde/coupling/fde_diabatic_postprocess.h"

#include "source_lcao/module_fde/coupling/fde_coupling_provider_factory.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fde
{

namespace
{

std::ifstream open_input(const std::string& path, const char* description)
{
    std::ifstream input(path.c_str());
    if (!input)
    {
        throw std::runtime_error(std::string("Cannot open FDE ") + description + ": " + path);
    }
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

std::vector<double> read_ao_overlap(const std::string& path)
{
    std::ifstream input = open_input(path, "AO-overlap artifact");
    std::string token;
    int version = 0;
    if (!(input >> token >> version) || token != "FDE_AO_MATRIX" || version != 1)
    {
        throw std::invalid_argument("Unsupported FDE AO-overlap artifact header");
    }
    std::size_t dimension = 0;
    if (!(input >> token >> dimension) || token != "DIMENSION" || dimension == 0)
    {
        throw std::invalid_argument("Invalid FDE AO-overlap dimension");
    }
    std::size_t size = 0;
    if (!(input >> token >> size) || token != "VALUES" || size != dimension * dimension)
    {
        throw std::invalid_argument("Invalid FDE AO-overlap matrix size");
    }
    std::vector<double> overlap(size, 0.0);
    for (std::size_t index = 0; index < size; ++index)
    {
        input >> overlap[index];
    }
    if (!(input >> token) || token != "END" || !input)
    {
        throw std::invalid_argument("FDE AO-overlap artifact is truncated");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE AO-overlap artifact has trailing content");
    }
    for (std::size_t column = 0; column < dimension; ++column)
    {
        for (std::size_t row = 0; row < dimension; ++row)
        {
            if (!std::isfinite(overlap[row + column * dimension]))
            {
                throw std::invalid_argument("FDE AO-overlap matrix is non-finite");
            }
        }
    }
    return overlap;
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

void write_result(const std::string& prefix,
                  const DiabaticPostprocessResult& result)
{
    const std::string text_path = prefix + ".fde_diabatic";
    std::ofstream output(text_path.c_str());
    if (!output)
    {
        throw std::runtime_error("Cannot create FDE diabatic result: " + text_path);
    }
    output << std::setprecision(17);
    output << "FDE_DIABATIC_RESULT 1\n";
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
        const DiabaticPairAssembly& pair = result.assembly.pairs[index];
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
        const DiabaticPairAssembly& pair = result.assembly.pairs[index];
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
}

} // namespace

DiabaticPostprocessResult DiabaticPostprocessor::evaluate(
    const FdeRuntimeConfig& config,
    const std::vector<DiabaticDeterminantArtifact>& determinants,
    const std::vector<LinearizedStateArtifact>& linearized_states,
    const std::vector<double>& ao_overlap)
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
        approximation.state_indices.push_back(
            FdeRuntimeConfigIO::state_index(config, selected_states[selected]));
    }
    approximation.overlap_active_fragments
        = config.l_fragment_labels.empty() ? all_fragment_labels(config)
                                           : config.l_fragment_labels;
    approximation.determinant_fragments
        = config.m_fragment_labels.empty() ? all_fragment_labels(config)
                                           : config.m_fragment_labels;

    const std::unique_ptr<TransitionEnergyProvider> transition_energy
        = FdeCouplingProviderFactory::create(config.coupling_provider,
                                             linearized_states,
                                             ao_overlap,
                                             config.singular_value_tolerance);
    CouplingValidationControls coupling_validation;
    coupling_validation.overlap_reciprocity_tolerance
        = config.symmetry_tolerance;
    coupling_validation.transition_density_trace_tolerance
        = config.transition_density_trace_tolerance;
    DiabaticPostprocessResult result;
    result.assembly = FdeDiabaticAssembler::assemble(determinants,
                                                     diagonal_energies,
                                                     ao_overlap,
                                                     approximation,
                                                     *transition_energy,
                                                     coupling_validation,
                                                     config.singular_value_tolerance);
    NonorthogonalSolverControls solver_controls;
    solver_controls.overlap_eigenvalue_cutoff = config.overlap_eigenvalue_cutoff;
    solver_controls.symmetry_tolerance = config.symmetry_tolerance;
    solver_controls.residual_tolerance = config.residual_tolerance;
    result.adiabatic_solution
        = NonorthogonalMultistateSolver::solve(result.assembly.problem, solver_controls);
    for (std::size_t index = 0; index < result.assembly.pairs.size(); ++index)
    {
        const DiabaticPairAssembly& pair = result.assembly.pairs[index];
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
    return result;
}

DiabaticPostprocessResult DiabaticPostprocessor::run(const std::string& config_path)
{
    std::ifstream config_input = open_input(config_path, "runtime config");
    const FdeRuntimeConfig config = FdeRuntimeConfigIO::read(config_input);
    if (config.ao_overlap_path.empty())
    {
        throw std::invalid_argument("FDE diabatic postprocess requires AO_OVERLAP");
    }
    const std::vector<double> overlap = read_ao_overlap(config.ao_overlap_path);
    std::vector<DiabaticDeterminantArtifact> determinants;
    std::vector<LinearizedStateArtifact> linearized_states;
    for (std::size_t index = 0; index < config.states.size(); ++index)
    {
        const std::string& label = config.states[index].state.label;
        std::ifstream determinant_input
            = open_input(artifact_path(config.determinant_artifacts,
                                       label,
                                       "determinant artifact"),
                         "determinant artifact");
        determinants.push_back(DeterminantArtifactIO::read(determinant_input,
                                                           overlap,
                                                           1.0e-8));
        std::ifstream linear_input
            = open_input(artifact_path(config.linearized_state_artifacts,
                                       label,
                                       "linearized-state artifact"),
                         "linearized-state artifact");
        linearized_states.push_back(LinearizedStateArtifactIO::read(
            linear_input,
            config.symmetry_tolerance));
    }
    const DiabaticPostprocessResult result
        = DiabaticPostprocessor::evaluate(config,
                                          determinants,
                                          linearized_states,
                                          overlap);
    write_result(config.output_prefix, result);
    return result;
}

} // namespace fde
