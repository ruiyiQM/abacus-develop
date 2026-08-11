#ifndef FDE_RUNTIME_STATE_H
#define FDE_RUNTIME_STATE_H

#include <cstddef>
#include <string>
#include <vector>

namespace fde
{

struct FragmentDefinition
{
    std::string label;
    std::vector<std::size_t> atom_indices;
    int neutral_valence_electrons;
};

struct FragmentChargeSpin
{
    std::string fragment_label;
    int charge;
    int spin_projection;
};

struct DiabaticStateSpec
{
    std::string label;
    std::vector<FragmentChargeSpin> fragments;
};

struct SpinPopulation
{
    int alpha;
    int beta;
};

class StateDefinition
{
  public:
    static void validate_partition(const std::vector<FragmentDefinition>& fragments,
                                   const std::size_t atom_count);

    static void validate_state(const std::vector<FragmentDefinition>& fragments,
                               const DiabaticStateSpec& state,
                               const int total_charge,
                               const int total_spin_projection);

    static SpinPopulation spin_population(const FragmentDefinition& fragment,
                                          const FragmentChargeSpin& assignment);
};

} // namespace fde

#endif // FDE_RUNTIME_STATE_H
