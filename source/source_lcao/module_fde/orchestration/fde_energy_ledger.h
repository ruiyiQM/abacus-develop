#ifndef FDE_ORCHESTRATION_ENERGY_LEDGER_H
#define FDE_ORCHESTRATION_ENERGY_LEDGER_H

#include <iosfwd>

namespace fde
{

struct EnergyLedger
{
    double orbital_kinetic_a_ry;
    double orbital_kinetic_b_ry;
    double nonadditive_kinetic_ry;
    double total_hartree_ry;
    double total_xc_ry;
    double local_external_ry;
    double nonlocal_external_a_ry;
    double nonlocal_external_b_ry;
    double ion_ion_ry;
};

class CanonicalEnergy
{
  public:
    static void validate(const EnergyLedger& ledger);
    static double total(const EnergyLedger& ledger);
    static void write(std::ostream& output, const EnergyLedger& ledger);
    static EnergyLedger read(std::istream& input);
};

} // namespace fde

#endif // FDE_ORCHESTRATION_ENERGY_LEDGER_H
