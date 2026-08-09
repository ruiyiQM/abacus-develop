#ifndef FDE_LCAO_DRIVER_H
#define FDE_LCAO_DRIVER_H

#include "fde_density_artifact.h"
#include "fde_projected_hamiltonian.h"
#include "fde_runtime_config.h"
#include "source_base/module_device/types.h"

#include <cstddef>
#include <complex>
#include <memory>
#include <string>
#include <vector>

class Charge;
struct Input_para;
class Parallel_Orbitals;
class UnitCell;
class K_Vectors;

namespace ModulePW
{
class PW_Basis;
}

namespace elecstate
{
class ElecState;
class Potential;
template <typename TK, typename TR>
class DensityMatrix;
}

namespace hamilt
{
template <typename T, typename Device>
class Hamilt;
}

namespace psi
{
template <typename T, typename Device>
class Psi;
}

namespace fde
{

class PotFde;

struct FdeScfStatus
{
    bool converged;
    int iterations;
    double density_residual;
};

/** Runtime-owned bridge from one task-local FDE_CONFIG to ABACUS LCAO. */
class FdeLcaoDriver
{
  public:
    static std::unique_ptr<FdeLcaoDriver> create(const Input_para& input,
                                                 const UnitCell& unit_cell,
                                                 const Parallel_Orbitals& orbitals);

    const FdeRuntimeConfig& config() const;
    const std::vector<std::size_t>& active_orbitals() const;
    int active_alpha_electrons() const;
    int active_beta_electrons() const;

    void initialize_active_charge(Charge& charge) const;
    void validate_core_density(const Charge& charge) const;
    void attach_embedding_potential(ModulePW::PW_Basis& density_basis,
                                    const UnitCell& unit_cell,
                                    elecstate::Potential& potential);

    std::unique_ptr<FdeProjectedHamiltonian>
    projected_hamiltonian(
        hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian,
        const Parallel_Orbitals& orbitals,
        std::size_t spin_kpoint_count) const;

    std::unique_ptr<FdeProjectedHamiltonianComplex>
    projected_hamiltonian(
        hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>&
            full_hamiltonian,
        const Parallel_Orbitals& orbitals,
        std::size_t spin_kpoint_count) const;

    void solve_projected(hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian,
                         psi::Psi<double, base_device::DEVICE_CPU>& wavefunctions,
                         elecstate::ElecState& electronic_state,
                         elecstate::DensityMatrix<double, double>& density_matrix,
                         Charge& charge,
                         const Parallel_Orbitals& orbitals) const;

    void solve_projected(
        hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>&
            full_hamiltonian,
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& wavefunctions,
        elecstate::ElecState& electronic_state,
        elecstate::DensityMatrix<std::complex<double>, double>& density_matrix,
        Charge& charge,
        const Parallel_Orbitals& orbitals) const;

    void write_scf_artifacts(
        Charge& charge,
        psi::Psi<double, base_device::DEVICE_CPU>& wavefunctions,
        elecstate::ElecState& electronic_state,
        hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian,
        const K_Vectors& kpoints,
        const Parallel_Orbitals& orbitals,
        const FdeScfStatus& status);

    void write_scf_artifacts(
        Charge& charge,
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& wavefunctions,
        elecstate::ElecState& electronic_state,
        hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>&
            full_hamiltonian,
        const K_Vectors& kpoints,
        const Parallel_Orbitals& orbitals,
        const FdeScfStatus& status);

  private:
    FdeLcaoDriver(const FdeRuntimeConfig& config,
                  const std::vector<std::size_t>& active_orbitals,
                  std::size_t full_ao_dimension,
                  const std::string& ks_solver,
                  int kpar,
                  int nbands,
                  double electron_count,
                  int active_alpha_electrons,
                  int active_beta_electrons,
                  const FrozenDensityArtifact& active_initial,
                  const std::vector<FrozenDensityArtifact>& frozen_environment);

    FdeRuntimeConfig config_;
    std::vector<std::size_t> active_orbitals_;
    std::size_t full_ao_dimension_;
    std::string ks_solver_;
    int kpar_;
    int nbands_;
    double electron_count_;
    int active_alpha_electrons_;
    int active_beta_electrons_;
    FrozenDensityArtifact active_initial_;
    std::vector<FrozenDensityArtifact> frozen_environment_;
    std::vector<double> active_alpha_local_;
    std::vector<double> active_beta_local_;
    std::vector<double> frozen_alpha_local_;
    std::vector<double> frozen_beta_local_;
    ModulePW::PW_Basis* density_basis_;
    PotFde* embedding_potential_;
};

} // namespace fde

#endif // FDE_LCAO_DRIVER_H
