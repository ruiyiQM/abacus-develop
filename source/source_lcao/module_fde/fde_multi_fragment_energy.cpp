#include "fde_multi_fragment_energy.h"

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
        throw std::invalid_argument(std::string("FDE multi-fragment energy expected token ")
                                    + expected);
    }
}

bool valid_label(const std::string& label)
{
    return !label.empty() && label.find_first_of(" \t\r\n") == std::string::npos;
}

} // namespace

void MultiFragmentCanonicalEnergy::validate(const MultiFragmentEnergyLedger& ledger)
{
    if (ledger.fragments.size() < 2)
    {
        throw std::invalid_argument("FDE multi-fragment energy requires at least two fragments");
    }
    std::set<std::string> labels;
    for (std::size_t index = 0; index < ledger.fragments.size(); ++index)
    {
        const FragmentEnergyTerm& fragment = ledger.fragments[index];
        if (!valid_label(fragment.fragment_label)
            || !labels.insert(fragment.fragment_label).second
            || !std::isfinite(fragment.orbital_kinetic_ry)
            || !std::isfinite(fragment.nonlocal_external_ry))
        {
            throw std::invalid_argument("FDE multi-fragment energy has an invalid fragment term");
        }
    }
    const double shared_terms[] = {ledger.nonadditive_kinetic_ry,
                                   ledger.total_hartree_ry,
                                   ledger.total_xc_ry,
                                   ledger.local_external_ry,
                                   ledger.ion_ion_ry};
    for (std::size_t index = 0; index < sizeof(shared_terms) / sizeof(shared_terms[0]); ++index)
    {
        if (!std::isfinite(shared_terms[index]))
        {
            throw std::invalid_argument("FDE multi-fragment energy has a non-finite shared term");
        }
    }
}

double MultiFragmentCanonicalEnergy::total(const MultiFragmentEnergyLedger& ledger)
{
    MultiFragmentCanonicalEnergy::validate(ledger);
    double result = ledger.nonadditive_kinetic_ry + ledger.total_hartree_ry
                    + ledger.total_xc_ry + ledger.local_external_ry + ledger.ion_ion_ry;
    for (std::size_t index = 0; index < ledger.fragments.size(); ++index)
    {
        result += ledger.fragments[index].orbital_kinetic_ry
                  + ledger.fragments[index].nonlocal_external_ry;
    }
    return result;
}

void MultiFragmentCanonicalEnergy::write(std::ostream& output,
                                         const MultiFragmentEnergyLedger& ledger)
{
    MultiFragmentCanonicalEnergy::validate(ledger);
    output << std::setprecision(17);
    output << "FDE_MULTI_FRAGMENT_ENERGY_LEDGER 1\n";
    output << "FRAGMENTS " << ledger.fragments.size() << '\n';
    for (std::size_t index = 0; index < ledger.fragments.size(); ++index)
    {
        const FragmentEnergyTerm& fragment = ledger.fragments[index];
        output << "FRAGMENT " << fragment.fragment_label << ' '
               << fragment.orbital_kinetic_ry << ' '
               << fragment.nonlocal_external_ry << '\n';
    }
    output << "NONADDITIVE_KINETIC_RY " << ledger.nonadditive_kinetic_ry << '\n';
    output << "TOTAL_HARTREE_RY " << ledger.total_hartree_ry << '\n';
    output << "TOTAL_XC_RY " << ledger.total_xc_ry << '\n';
    output << "LOCAL_EXTERNAL_RY " << ledger.local_external_ry << '\n';
    output << "ION_ION_RY " << ledger.ion_ion_ry << '\n';
    output << "TOTAL_RY " << MultiFragmentCanonicalEnergy::total(ledger) << "\nEND\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write FDE multi-fragment energy ledger");
    }
}

MultiFragmentEnergyLedger MultiFragmentCanonicalEnergy::read(std::istream& input)
{
    require_token(input, "FDE_MULTI_FRAGMENT_ENERGY_LEDGER");
    int version = 0;
    input >> version;
    if (version != 1)
    {
        throw std::invalid_argument("Unsupported FDE multi-fragment energy ledger version");
    }
    require_token(input, "FRAGMENTS");
    std::size_t fragment_count = 0;
    input >> fragment_count;
    MultiFragmentEnergyLedger ledger;
    ledger.fragments.resize(fragment_count);
    for (std::size_t index = 0; index < fragment_count; ++index)
    {
        require_token(input, "FRAGMENT");
        input >> ledger.fragments[index].fragment_label
              >> ledger.fragments[index].orbital_kinetic_ry
              >> ledger.fragments[index].nonlocal_external_ry;
    }
    require_token(input, "NONADDITIVE_KINETIC_RY");
    input >> ledger.nonadditive_kinetic_ry;
    require_token(input, "TOTAL_HARTREE_RY");
    input >> ledger.total_hartree_ry;
    require_token(input, "TOTAL_XC_RY");
    input >> ledger.total_xc_ry;
    require_token(input, "LOCAL_EXTERNAL_RY");
    input >> ledger.local_external_ry;
    require_token(input, "ION_ION_RY");
    input >> ledger.ion_ion_ry;
    require_token(input, "TOTAL_RY");
    double recorded_total = 0.0;
    input >> recorded_total;
    require_token(input, "END");
    if (!input)
    {
        throw std::invalid_argument("FDE multi-fragment energy ledger is truncated");
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::invalid_argument("FDE multi-fragment energy ledger has trailing content");
    }

    MultiFragmentCanonicalEnergy::validate(ledger);
    const double recomputed = MultiFragmentCanonicalEnergy::total(ledger);
    const double tolerance = 1.0e-12 * std::max(1.0, std::fabs(recomputed));
    if (!std::isfinite(recorded_total) || std::fabs(recorded_total - recomputed) > tolerance)
    {
        throw std::invalid_argument("FDE multi-fragment energy total is inconsistent");
    }
    return ledger;
}

} // namespace fde
