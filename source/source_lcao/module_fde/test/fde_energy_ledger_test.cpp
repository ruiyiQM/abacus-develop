#include "../orchestration/fde_energy_ledger.h"

#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{

fde::EnergyLedger ledger()
{
    return {1.0, 2.0, 0.3, 4.0, -1.0, -5.0, 0.2, 0.4, 3.0};
}

} // namespace

TEST(FdeEnergyLedger, SumsEveryCanonicalTermOnce)
{
    EXPECT_DOUBLE_EQ(fde::CanonicalEnergy::total(ledger()), 4.9);
}

TEST(FdeEnergyLedger, RoundTripsAndChecksRecordedTotal)
{
    std::ostringstream output;
    fde::CanonicalEnergy::write(output, ledger());
    std::istringstream input(output.str());
    const fde::EnergyLedger restored = fde::CanonicalEnergy::read(input);
    EXPECT_DOUBLE_EQ(fde::CanonicalEnergy::total(restored), 4.9);

    std::string corrupted = output.str();
    const std::size_t total_position = corrupted.find("TOTAL_RY 4.9000000000000004");
    ASSERT_NE(total_position, std::string::npos);
    corrupted.replace(total_position, std::string("TOTAL_RY 4.9000000000000004").size(), "TOTAL_RY 9");
    std::istringstream corrupted_input(corrupted);
    EXPECT_THROW(fde::CanonicalEnergy::read(corrupted_input), std::invalid_argument);
}

TEST(FdeEnergyLedger, RejectsNonfiniteTerms)
{
    fde::EnergyLedger invalid = ledger();
    invalid.total_xc_ry = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(fde::CanonicalEnergy::total(invalid), std::invalid_argument);
}
