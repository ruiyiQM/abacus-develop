#include "fde_state.h"

#include <set>
#include <stdexcept>

namespace fde
{

namespace
{

void validate_label(const std::string& label, const char* description)
{
    if (label.empty() || label.find_first_of(" \t\r\n") != std::string::npos)
    {
        throw std::invalid_argument(std::string("FDE ") + description
                                    + " must be nonempty and contain no whitespace");
    }
}

const FragmentDefinition& find_fragment(const std::vector<FragmentDefinition>& fragments,
                                        const std::string& label)
{
    for (std::size_t index = 0; index < fragments.size(); ++index)
    {
        if (fragments[index].label == label)
        {
            return fragments[index];
        }
    }
    throw std::invalid_argument("FDE state refers to an unknown fragment: " + label);
}

} // namespace

void StateDefinition::validate_partition(const std::vector<FragmentDefinition>& fragments,
                                         const std::size_t atom_count)
{
    if (fragments.size() != 2)
    {
        throw std::invalid_argument("FDE RP1 requires exactly two fragments");
    }
    if (atom_count == 0)
    {
        throw std::invalid_argument("FDE supersystem must contain at least one atom");
    }

    std::set<std::string> labels;
    std::vector<bool> assigned(atom_count, false);
    for (std::size_t fragment_index = 0; fragment_index < fragments.size(); ++fragment_index)
    {
        const FragmentDefinition& fragment = fragments[fragment_index];
        validate_label(fragment.label, "fragment label");
        if (!labels.insert(fragment.label).second)
        {
            throw std::invalid_argument("FDE fragment labels must be unique");
        }
        if (fragment.atom_indices.empty())
        {
            throw std::invalid_argument("FDE fragment atom list must not be empty");
        }
        if (fragment.neutral_valence_electrons < 0)
        {
            throw std::invalid_argument("FDE neutral valence-electron count must be nonnegative");
        }
        for (std::size_t atom_position = 0; atom_position < fragment.atom_indices.size(); ++atom_position)
        {
            const std::size_t atom_index = fragment.atom_indices[atom_position];
            if (atom_index >= atom_count)
            {
                throw std::out_of_range("FDE fragment atom index is outside the supersystem");
            }
            if (assigned[atom_index])
            {
                throw std::invalid_argument("FDE fragments must not share atoms");
            }
            assigned[atom_index] = true;
        }
    }

    for (std::size_t atom_index = 0; atom_index < assigned.size(); ++atom_index)
    {
        if (!assigned[atom_index])
        {
            throw std::invalid_argument("FDE fragment partition must cover every supersystem atom");
        }
    }
}

SpinPopulation StateDefinition::spin_population(const FragmentDefinition& fragment,
                                                const FragmentChargeSpin& assignment)
{
    if (fragment.label != assignment.fragment_label)
    {
        throw std::invalid_argument("FDE charge/spin assignment does not match its fragment");
    }
    const int electrons = fragment.neutral_valence_electrons - assignment.charge;
    if (electrons < 0 || (electrons + assignment.spin_projection) % 2 != 0)
    {
        throw std::invalid_argument("FDE fragment electron count and spin projection have invalid parity");
    }

    const SpinPopulation population{
        (electrons + assignment.spin_projection) / 2,
        (electrons - assignment.spin_projection) / 2};
    if (population.alpha < 0 || population.beta < 0)
    {
        throw std::invalid_argument("FDE fragment spin populations must be nonnegative");
    }
    return population;
}

void StateDefinition::validate_state(const std::vector<FragmentDefinition>& fragments,
                                     const DiabaticStateSpec& state,
                                     const int total_charge,
                                     const int total_spin_projection)
{
    validate_label(state.label, "state label");
    if (state.fragments.size() != fragments.size())
    {
        throw std::invalid_argument("FDE state must assign charge and spin to every fragment");
    }

    std::set<std::string> assigned_labels;
    int charge_sum = 0;
    int spin_sum = 0;
    for (std::size_t index = 0; index < state.fragments.size(); ++index)
    {
        const FragmentChargeSpin& assignment = state.fragments[index];
        validate_label(assignment.fragment_label, "state fragment label");
        if (!assigned_labels.insert(assignment.fragment_label).second)
        {
            throw std::invalid_argument("FDE state assigns a fragment more than once");
        }
        const FragmentDefinition& fragment = find_fragment(fragments, assignment.fragment_label);
        StateDefinition::spin_population(fragment, assignment);
        charge_sum += assignment.charge;
        spin_sum += assignment.spin_projection;
    }

    if (charge_sum != total_charge)
    {
        throw std::invalid_argument("FDE fragment charges do not reproduce the supersystem charge");
    }
    if (spin_sum != total_spin_projection)
    {
        throw std::invalid_argument("FDE fragment spins do not reproduce the supersystem spin projection");
    }
}

} // namespace fde
