#ifndef FDE_ORCHESTRATION_MULTI_FRAGMENT_ENERGY_H
#define FDE_ORCHESTRATION_MULTI_FRAGMENT_ENERGY_H

#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

struct FragmentEnergyTerm
{
    std::string fragment_label;
    double orbital_kinetic_ry;
    double nonlocal_external_ry;
};

struct MultiFragmentEnergyLedger
{
    std::vector<FragmentEnergyTerm> fragments;
    double nonadditive_kinetic_ry;
    double total_hartree_ry;
    double total_xc_ry;
    double local_external_ry;
    double ion_ion_ry;
};

class MultiFragmentCanonicalEnergy
{
  public:
    static void validate(const MultiFragmentEnergyLedger& ledger);
    static double total(const MultiFragmentEnergyLedger& ledger);
    static void write(std::ostream& output, const MultiFragmentEnergyLedger& ledger);
    static MultiFragmentEnergyLedger read(std::istream& input);
};

} // namespace fde

#endif // FDE_ORCHESTRATION_MULTI_FRAGMENT_ENERGY_H
