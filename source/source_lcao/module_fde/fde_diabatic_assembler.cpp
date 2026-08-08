#include "fde_diabatic_assembler.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fde
{

namespace
{

void validate_token(const std::string& value, const char* description)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(std::string("FDE-diab ") + description
                                    + " must be a nonempty token");
    }
}

std::set<std::string> checked_label_set(const std::vector<std::string>& labels,
                                       const char* description)
{
    std::set<std::string> result;
    for (std::size_t index = 0; index < labels.size(); ++index)
    {
        validate_token(labels[index], description);
        if (!result.insert(labels[index]).second)
        {
            throw std::invalid_argument(std::string("FDE-diab duplicate ") + description);
        }
    }
    return result;
}

std::set<std::string> available_fragments(
    const std::vector<DiabaticDeterminantArtifact>& determinants)
{
    std::set<std::string> result;
    for (std::size_t state = 0; state < determinants.size(); ++state)
    {
        result.insert(determinants[state].alpha.source_fragment_labels.begin(),
                      determinants[state].alpha.source_fragment_labels.end());
        result.insert(determinants[state].beta.source_fragment_labels.begin(),
                      determinants[state].beta.source_fragment_labels.end());
    }
    return result;
}

OccupiedSpinOrbitals select_spin_orbitals(const OccupiedSpinOrbitals& input,
                                          const std::size_t ao_dimension,
                                          const std::set<std::string>& selected_fragments)
{
    const std::size_t count
        = DeterminantArtifactIO::occupied_count(input, ao_dimension);
    OccupiedSpinOrbitals result;
    for (std::size_t orbital = 0; orbital < count; ++orbital)
    {
        if (selected_fragments.count(input.source_fragment_labels[orbital]) == 0)
        {
            continue;
        }
        const std::size_t target_offset = result.coefficients.size();
        result.coefficients.resize(target_offset + ao_dimension);
        for (std::size_t ao = 0; ao < ao_dimension; ++ao)
        {
            result.coefficients[target_offset + ao]
                = input.coefficients[ao + orbital * ao_dimension];
        }
        result.orbital_energies_ry.push_back(input.orbital_energies_ry[orbital]);
        result.source_fragment_labels.push_back(input.source_fragment_labels[orbital]);
    }
    return result;
}

DiabaticDeterminantArtifact select_determinant_fragments(
    const DiabaticDeterminantArtifact& input,
    const std::set<std::string>& selected_fragments,
    const std::vector<double>& ao_overlap)
{
    DiabaticDeterminantArtifact result = input;
    result.alpha = select_spin_orbitals(input.alpha,
                                        input.ao_dimension,
                                        selected_fragments);
    result.beta = select_spin_orbitals(input.beta,
                                       input.ao_dimension,
                                       selected_fragments);
    DeterminantArtifactIO::validate(result, ao_overlap, 1.0e-8);
    return result;
}

void validate_electron_counts(const std::vector<DiabaticDeterminantArtifact>& determinants)
{
    const std::size_t dimension = determinants[0].ao_dimension;
    const std::size_t alpha
        = DeterminantArtifactIO::occupied_count(determinants[0].alpha, dimension);
    const std::size_t beta
        = DeterminantArtifactIO::occupied_count(determinants[0].beta, dimension);
    for (std::size_t state = 1; state < determinants.size(); ++state)
    {
        if (DeterminantArtifactIO::occupied_count(determinants[state].alpha, dimension)
                != alpha
            || DeterminantArtifactIO::occupied_count(determinants[state].beta, dimension)
                   != beta)
        {
            throw std::invalid_argument(
                "FDE-diab M-subsystem selection changes electron count between states");
        }
    }
}

} // namespace

FdeDiabFragmentOverlapPolicy::FdeDiabFragmentOverlapPolicy(
    const std::vector<std::string>& overlap_active_fragments)
    : overlap_active_fragments_(overlap_active_fragments)
{
    checked_label_set(overlap_active_fragments_, "L-fragment label");
}

bool FdeDiabFragmentOverlapPolicy::retain(const std::string& bra_fragment,
                                          const std::string& ket_fragment) const
{
    if (bra_fragment == ket_fragment)
    {
        return true;
    }
    return std::find(overlap_active_fragments_.begin(),
                     overlap_active_fragments_.end(),
                     bra_fragment)
               != overlap_active_fragments_.end()
           || std::find(overlap_active_fragments_.begin(),
                        overlap_active_fragments_.end(),
                        ket_fragment)
                  != overlap_active_fragments_.end();
}

FdeDiabaticAssemblyResult FdeDiabaticAssembler::assemble(
    const std::vector<DiabaticDeterminantArtifact>& determinants,
    const std::vector<double>& diagonal_energies_ry,
    const std::vector<double>& ao_overlap,
    const FdeDiabApproximationSpec& approximation,
    const TransitionEnergyProvider& transition_energy_provider,
    const double singular_value_tolerance)
{
    if (determinants.size() < 2 || diagonal_energies_ry.size() != determinants.size()
        || approximation.state_indices.size() < 2
        || approximation.determinant_fragments.empty()
        || !std::isfinite(singular_value_tolerance) || singular_value_tolerance <= 0.0)
    {
        throw std::invalid_argument("FDE-diab(K,L,M) assembly inputs are invalid");
    }
    for (std::size_t state = 0; state < determinants.size(); ++state)
    {
        if (!std::isfinite(diagonal_energies_ry[state]))
        {
            throw std::invalid_argument("FDE-diab diagonal energy is non-finite");
        }
        DeterminantArtifactIO::validate(determinants[state], ao_overlap, 1.0e-8);
    }

    std::set<std::size_t> selected_states;
    for (std::size_t index = 0; index < approximation.state_indices.size(); ++index)
    {
        const std::size_t state = approximation.state_indices[index];
        if (state >= determinants.size() || !selected_states.insert(state).second)
        {
            throw std::invalid_argument("FDE-diab K-state indices must be unique and in range");
        }
    }
    const std::set<std::string> selected_fragments
        = checked_label_set(approximation.determinant_fragments, "M-fragment label");
    const std::set<std::string> overlap_active
        = checked_label_set(approximation.overlap_active_fragments, "L-fragment label");
    const std::set<std::string> available = available_fragments(determinants);
    for (std::set<std::string>::const_iterator fragment = selected_fragments.begin();
         fragment != selected_fragments.end();
         ++fragment)
    {
        if (available.count(*fragment) == 0)
        {
            throw std::invalid_argument("FDE-diab M selection refers to an unavailable fragment");
        }
    }
    for (std::set<std::string>::const_iterator fragment = overlap_active.begin();
         fragment != overlap_active.end();
         ++fragment)
    {
        if (selected_fragments.count(*fragment) == 0)
        {
            throw std::invalid_argument("FDE-diab L fragments must be a subset of M fragments");
        }
    }

    std::vector<DiabaticDeterminantArtifact> selected_determinants;
    selected_determinants.reserve(approximation.state_indices.size());
    for (std::size_t index = 0; index < approximation.state_indices.size(); ++index)
    {
        selected_determinants.push_back(select_determinant_fragments(
            determinants[approximation.state_indices[index]],
            selected_fragments,
            ao_overlap));
    }
    validate_electron_counts(selected_determinants);

    FdeDiabaticAssemblyResult result;
    result.audit.k_states = approximation.state_indices.size();
    result.audit.l_overlap_active_fragments = approximation.overlap_active_fragments.size();
    result.audit.m_determinant_fragments = approximation.determinant_fragments.size();
    for (std::size_t state = 0; state < determinants.size(); ++state)
    {
        if (selected_states.count(state) == 0)
        {
            result.audit.omitted_state_indices.push_back(state);
        }
    }
    for (std::set<std::string>::const_iterator fragment = available.begin();
         fragment != available.end();
         ++fragment)
    {
        if (selected_fragments.count(*fragment) == 0)
        {
            result.audit.omitted_determinant_fragments.push_back(*fragment);
        }
    }
    result.audit.retained_interfragment_overlap_pairs = 0;
    result.audit.discarded_interfragment_overlap_pairs = 0;
    for (std::size_t first = 0; first < approximation.determinant_fragments.size(); ++first)
    {
        for (std::size_t second = first + 1;
             second < approximation.determinant_fragments.size();
             ++second)
        {
            if (overlap_active.count(approximation.determinant_fragments[first]) > 0
                || overlap_active.count(approximation.determinant_fragments[second]) > 0)
            {
                ++result.audit.retained_interfragment_overlap_pairs;
            }
            else
            {
                ++result.audit.discarded_interfragment_overlap_pairs;
            }
        }
    }

    const std::size_t k = approximation.state_indices.size();
    result.problem.hamiltonian_ry.assign(k * k, 0.0);
    result.problem.overlap.assign(k * k, 0.0);
    std::set<std::string> selected_state_labels;
    for (std::size_t state = 0; state < k; ++state)
    {
        const std::size_t input_state = approximation.state_indices[state];
        if (!selected_state_labels.insert(determinants[input_state].state_label).second)
        {
            throw std::invalid_argument("FDE-diab selected state labels must be unique");
        }
        result.problem.state_labels.push_back(determinants[input_state].state_label);
        result.problem.hamiltonian_ry[state + state * k]
            = diagonal_energies_ry[input_state];
        result.problem.overlap[state + state * k] = 1.0;
    }

    const FdeDiabFragmentOverlapPolicy overlap_policy(
        approximation.overlap_active_fragments);
    for (std::size_t second = 1; second < k; ++second)
    {
        for (std::size_t first = 0; first < second; ++first)
        {
            const ElectronicCouplingResult coupling
                = ElectronicCoupling::evaluate_symmetric_with_policy(
                    selected_determinants[first],
                    selected_determinants[second],
                    ao_overlap,
                    overlap_policy,
                    transition_energy_provider,
                    singular_value_tolerance);
            result.problem.overlap[first + second * k] = coupling.normalized_overlap;
            result.problem.overlap[second + first * k] = coupling.normalized_overlap;
            result.problem.hamiltonian_ry[first + second * k]
                = coupling.hamiltonian_coupling_ry;
            result.problem.hamiltonian_ry[second + first * k]
                = coupling.hamiltonian_coupling_ry;
            result.pairs.push_back({approximation.state_indices[first],
                                    approximation.state_indices[second],
                                    coupling.normalized_overlap,
                                    coupling.hamiltonian_coupling_ry});
        }
    }
    return result;
}

} // namespace fde
