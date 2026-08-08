#include "../fde_state.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace
{

std::vector<fde::FragmentDefinition> reaction_fragments()
{
    return {
        {"F", {0}, 7},
        {"CH3Cl", {1, 2, 3, 4, 5}, 14}};
}

} // namespace

TEST(FdeState, ValidatesChargeLocalizedReactionStates)
{
    const std::vector<fde::FragmentDefinition> fragments = reaction_fragments();
    fde::StateDefinition::validate_partition(fragments, 6);

    const fde::DiabaticStateSpec reactant{
        "F-_CH3Cl", {{"F", -1, 0}, {"CH3Cl", 0, 0}}};
    const fde::DiabaticStateSpec product{
        "F_CH3Cl-", {{"F", 0, 1}, {"CH3Cl", -1, -1}}};

    EXPECT_NO_THROW(fde::StateDefinition::validate_state(fragments, reactant, -1, 0));
    EXPECT_NO_THROW(fde::StateDefinition::validate_state(fragments, product, -1, 0));

    const fde::SpinPopulation fluorine
        = fde::StateDefinition::spin_population(fragments[0], product.fragments[0]);
    const fde::SpinPopulation methyl_chloride
        = fde::StateDefinition::spin_population(fragments[1], product.fragments[1]);
    EXPECT_EQ(fluorine.alpha, 4);
    EXPECT_EQ(fluorine.beta, 3);
    EXPECT_EQ(methyl_chloride.alpha, 7);
    EXPECT_EQ(methyl_chloride.beta, 8);
}

TEST(FdeState, RejectsInvalidPartitionsAndElectronicAssignments)
{
    EXPECT_THROW(fde::StateDefinition::validate_partition(
                     {{"A", {0, 1}, 2}, {"B", {1}, 1}}, 2),
                 std::invalid_argument);
    EXPECT_THROW(fde::StateDefinition::validate_partition(
                     {{"A", {0}, 1}, {"B", {2}, 1}}, 3),
                 std::invalid_argument);

    const std::vector<fde::FragmentDefinition> fragments = reaction_fragments();
    EXPECT_THROW(fde::StateDefinition::validate_state(
                     fragments,
                     {"bad_charge", {{"F", 0, 1}, {"CH3Cl", 0, -1}}},
                     -1,
                     0),
                 std::invalid_argument);
    EXPECT_THROW(fde::StateDefinition::validate_state(
                     fragments,
                     {"bad_parity", {{"F", 0, 0}, {"CH3Cl", -1, 0}}},
                     -1,
                     0),
                 std::invalid_argument);
}
