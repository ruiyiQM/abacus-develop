#include "fde_energy_ledger.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>
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
        throw std::invalid_argument(std::string("FDE energy ledger expected token ") + expected);
    }
}

} // namespace

void CanonicalEnergy::validate(const EnergyLedger& ledger)
{
    const double values[] = {
        ledger.orbital_kinetic_a_ry,
        ledger.orbital_kinetic_b_ry,
        ledger.nonadditive_kinetic_ry,
        ledger.total_hartree_ry,
        ledger.total_xc_ry,
        ledger.local_external_ry,
        ledger.nonlocal_external_a_ry,
        ledger.nonlocal_external_b_ry,
        ledger.ion_ion_ry};
    for (std::size_t index = 0; index < sizeof(values) / sizeof(values[0]); ++index)
    {
        if (!std::isfinite(values[index]))
        {
            throw std::invalid_argument("FDE canonical energy ledger contains a non-finite term");
        }
    }
}

double CanonicalEnergy::total(const EnergyLedger& ledger)
{
    CanonicalEnergy::validate(ledger);
    return ledger.orbital_kinetic_a_ry + ledger.orbital_kinetic_b_ry
           + ledger.nonadditive_kinetic_ry + ledger.total_hartree_ry
           + ledger.total_xc_ry + ledger.local_external_ry
           + ledger.nonlocal_external_a_ry + ledger.nonlocal_external_b_ry
           + ledger.ion_ion_ry;
}

void CanonicalEnergy::write(std::ostream& output, const EnergyLedger& ledger)
{
    CanonicalEnergy::validate(ledger);
    output << std::setprecision(17);
    output << "FDE_ENERGY_LEDGER 1\n";
    output << "ORBITAL_KINETIC_A_RY " << ledger.orbital_kinetic_a_ry << '\n';
    output << "ORBITAL_KINETIC_B_RY " << ledger.orbital_kinetic_b_ry << '\n';
    output << "NONADDITIVE_KINETIC_RY " << ledger.nonadditive_kinetic_ry << '\n';
    output << "TOTAL_HARTREE_RY " << ledger.total_hartree_ry << '\n';
    output << "TOTAL_XC_RY " << ledger.total_xc_ry << '\n';
    output << "LOCAL_EXTERNAL_RY " << ledger.local_external_ry << '\n';
    output << "NONLOCAL_EXTERNAL_A_RY " << ledger.nonlocal_external_a_ry << '\n';
    output << "NONLOCAL_EXTERNAL_B_RY " << ledger.nonlocal_external_b_ry << '\n';
    output << "ION_ION_RY " << ledger.ion_ion_ry << '\n';
    output << "TOTAL_RY " << CanonicalEnergy::total(ledger) << "\nEND\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE energy ledger");
    }
}

EnergyLedger CanonicalEnergy::read(std::istream& input)
{
    require_token(input, "FDE_ENERGY_LEDGER");
    int version = 0;
    input >> version;
    if (version != 1)
    {
        throw std::invalid_argument("Unsupported FDE energy ledger version");
    }
    EnergyLedger ledger;
    require_token(input, "ORBITAL_KINETIC_A_RY");
    input >> ledger.orbital_kinetic_a_ry;
    require_token(input, "ORBITAL_KINETIC_B_RY");
    input >> ledger.orbital_kinetic_b_ry;
    require_token(input, "NONADDITIVE_KINETIC_RY");
    input >> ledger.nonadditive_kinetic_ry;
    require_token(input, "TOTAL_HARTREE_RY");
    input >> ledger.total_hartree_ry;
    require_token(input, "TOTAL_XC_RY");
    input >> ledger.total_xc_ry;
    require_token(input, "LOCAL_EXTERNAL_RY");
    input >> ledger.local_external_ry;
    require_token(input, "NONLOCAL_EXTERNAL_A_RY");
    input >> ledger.nonlocal_external_a_ry;
    require_token(input, "NONLOCAL_EXTERNAL_B_RY");
    input >> ledger.nonlocal_external_b_ry;
    require_token(input, "ION_ION_RY");
    input >> ledger.ion_ion_ry;
    require_token(input, "TOTAL_RY");
    double recorded_total = 0.0;
    input >> recorded_total;
    require_token(input, "END");
    if (!input)
    {
        throw std::invalid_argument("FDE energy ledger is truncated or malformed");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE energy ledger contains trailing content");
    }
    CanonicalEnergy::validate(ledger);
    const double recomputed_total = CanonicalEnergy::total(ledger);
    const double tolerance = 1.0e-12 * std::max(1.0, std::fabs(recomputed_total));
    if (!std::isfinite(recorded_total) || std::fabs(recorded_total - recomputed_total) > tolerance)
    {
        throw std::invalid_argument("FDE energy ledger recorded total is inconsistent with named terms");
    }
    return ledger;
}

} // namespace fde
