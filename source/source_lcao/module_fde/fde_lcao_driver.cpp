#include "fde_lcao_driver.h"

#include "fde_potential_evaluator.h"
#include "fde_projected_hamiltonian.h"
#include "pot_fde.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_charge/charge.h"
#include "source_estate/module_pot/H_Hartree_pw.h"
#include "source_estate/module_pot/potential_new.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_hsolver/hsolver_lcao.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace fde
{

namespace
{

FdeRuntimeConfig read_config_file(const std::string& path)
{
    std::ifstream input(path.c_str());
    if (!input)
    {
        throw std::runtime_error("Cannot open FDE_CONFIG file: " + path);
    }
    return FdeRuntimeConfigIO::read(input);
}

FrozenDensityArtifact read_density_file(const std::string& path,
                                        const double electron_tolerance)
{
    std::ifstream input(path.c_str());
    if (!input)
    {
        throw std::runtime_error("Cannot open FDE density artifact: " + path);
    }
    return DensityArtifactIO::read(input, electron_tolerance);
}

const RuntimeStateDefinition& active_state(const FdeRuntimeConfig& config)
{
    return config.states[FdeRuntimeConfigIO::state_index(config, config.active_state)];
}

const FragmentChargeSpin& active_assignment(const FdeRuntimeConfig& config)
{
    const RuntimeStateDefinition& state = active_state(config);
    for (std::size_t index = 0; index < state.state.fragments.size(); ++index)
    {
        if (state.state.fragments[index].fragment_label == config.active_fragment)
        {
            return state.state.fragments[index];
        }
    }
    throw std::invalid_argument("FDE active state does not assign the active fragment");
}

std::vector<std::size_t> active_ao_indices(const FdeRuntimeConfig& config,
                                           const UnitCell& unit_cell,
                                           const std::size_t full_dimension)
{
    const FragmentDefinition& fragment
        = config.fragments[FdeRuntimeConfigIO::fragment_index(config, config.active_fragment)];
    const int* first_orbital = unit_cell.get_iat2iwt();
    if (first_orbital == nullptr)
    {
        throw std::invalid_argument("FDE active AO projection requires UnitCell iat2iwt data");
    }
    std::vector<std::size_t> result;
    for (std::size_t atom_position = 0; atom_position < fragment.atom_indices.size(); ++atom_position)
    {
        const std::size_t atom = fragment.atom_indices[atom_position];
        const std::size_t begin = static_cast<std::size_t>(first_orbital[atom]);
        const std::size_t end = atom + 1 < static_cast<std::size_t>(unit_cell.nat)
                                    ? static_cast<std::size_t>(first_orbital[atom + 1])
                                    : full_dimension;
        if (begin >= end || end > full_dimension)
        {
            throw std::invalid_argument("FDE UnitCell atom-to-AO range is invalid");
        }
        for (std::size_t orbital = begin; orbital < end; ++orbital)
        {
            result.push_back(orbital);
        }
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
    {
        throw std::invalid_argument("FDE active fragment produces duplicate AO indices");
    }
    return result;
}

double vector_norm(const double x, const double y, const double z)
{
    return std::sqrt(x * x + y * y + z * z);
}

double dot(const double ax,
           const double ay,
           const double az,
           const double bx,
           const double by,
           const double bz)
{
    return ax * bx + ay * by + az * bz;
}

UniformGrid make_grid(const FrozenDensityArtifact& artifact,
                      const ModulePW::PW_Basis& basis,
                      const UnitCell& unit_cell)
{
    if (artifact.grid_x != static_cast<std::size_t>(basis.nx)
        || artifact.grid_y != static_cast<std::size_t>(basis.ny)
        || artifact.grid_z != static_cast<std::size_t>(basis.nz)
        || artifact.grid_x * artifact.grid_y * artifact.grid_z
               != static_cast<std::size_t>(basis.nrxx))
    {
        throw std::invalid_argument(
            "FDE native LCAO runtime currently requires a replicated density grid");
    }
    const ModuleBase::Matrix3& lattice = unit_cell.latvec;
    const double a = vector_norm(lattice.e11, lattice.e12, lattice.e13) * unit_cell.lat0;
    const double b = vector_norm(lattice.e21, lattice.e22, lattice.e23) * unit_cell.lat0;
    const double c = vector_norm(lattice.e31, lattice.e32, lattice.e33) * unit_cell.lat0;
    const double orthogonality_scale = std::max(a * b, std::max(a * c, b * c));
    const double ab = dot(lattice.e11, lattice.e12, lattice.e13,
                          lattice.e21, lattice.e22, lattice.e23)
                      * unit_cell.lat0 * unit_cell.lat0;
    const double ac = dot(lattice.e11, lattice.e12, lattice.e13,
                          lattice.e31, lattice.e32, lattice.e33)
                      * unit_cell.lat0 * unit_cell.lat0;
    const double bc = dot(lattice.e21, lattice.e22, lattice.e23,
                          lattice.e31, lattice.e32, lattice.e33)
                      * unit_cell.lat0 * unit_cell.lat0;
    if (orthogonality_scale <= 0.0 || std::fabs(ab) > 1.0e-10 * orthogonality_scale
        || std::fabs(ac) > 1.0e-10 * orthogonality_scale
        || std::fabs(bc) > 1.0e-10 * orthogonality_scale)
    {
        throw std::invalid_argument(
            "FDE LC94/PBE grid adapter currently requires an orthogonal molecular cell");
    }
    UniformGrid grid;
    grid.x = artifact.grid_x;
    grid.y = artifact.grid_y;
    grid.z = artifact.grid_z;
    grid.spacing_x_bohr = a / grid.x;
    grid.spacing_y_bohr = b / grid.y;
    grid.spacing_z_bohr = c / grid.z;
    return grid;
}

} // namespace

FdeLcaoDriver::FdeLcaoDriver(
    const FdeRuntimeConfig& config,
    const std::vector<std::size_t>& active_orbitals,
    const std::size_t full_ao_dimension,
    const std::string& ks_solver,
    const int kpar,
    const int nbands,
    const double electron_count,
    const int active_alpha_electrons,
    const int active_beta_electrons,
    const FrozenDensityArtifact& active_initial,
    const std::vector<FrozenDensityArtifact>& frozen_environment)
    : config_(config),
      active_orbitals_(active_orbitals),
      full_ao_dimension_(full_ao_dimension),
      ks_solver_(ks_solver),
      kpar_(kpar),
      nbands_(nbands),
      electron_count_(electron_count),
      active_alpha_electrons_(active_alpha_electrons),
      active_beta_electrons_(active_beta_electrons),
      active_initial_(active_initial),
      frozen_environment_(frozen_environment)
{
}

std::unique_ptr<FdeLcaoDriver> FdeLcaoDriver::create(
    const Input_para& input,
    const UnitCell& unit_cell,
    const Parallel_Orbitals& orbitals)
{
    if (input.fde_task != "embedded_scf")
    {
        return std::unique_ptr<FdeLcaoDriver>();
    }
    const FdeRuntimeConfig config = read_config_file(input.fde_config);
    if (config.active_state.empty() || config.active_fragment.empty()
        || config.active_density_path.empty())
    {
        throw std::invalid_argument(
            "FDE embedded_scf requires ACTIVE_STATE, ACTIVE_FRAGMENT, and ACTIVE_DENSITY");
    }
    if (config.atom_count != static_cast<std::size_t>(unit_cell.nat))
    {
        throw std::invalid_argument("FDE_CONFIG atom count does not match STRU");
    }
    const std::size_t full_dimension
        = static_cast<std::size_t>(orbitals.get_global_row_size());
    if (orbitals.get_row_size() != orbitals.get_global_row_size()
        || orbitals.get_col_size() != orbitals.get_global_col_size())
    {
        throw std::invalid_argument(
            "FDE embedded_scf currently requires replicated Gamma AO matrices");
    }
    const std::vector<std::size_t> active_orbitals
        = active_ao_indices(config, unit_cell, full_dimension);
    const FragmentDefinition& fragment
        = config.fragments[FdeRuntimeConfigIO::fragment_index(config, config.active_fragment)];
    const SpinPopulation population
        = StateDefinition::spin_population(fragment, active_assignment(config));
    const int expected_electrons = population.alpha + population.beta;
    const int expected_spin = population.alpha - population.beta;
    if (std::fabs(input.nelec - expected_electrons) > config.electron_tolerance
        || std::fabs(input.nupdown - expected_spin) > config.electron_tolerance)
    {
        throw std::invalid_argument(
            "FDE INPUT nelec/nupdown do not match the active fragment state assignment");
    }
    if (input.ks_solver != "lapack" || input.kpar != 1
        || input.nbands < std::max(population.alpha, population.beta)
        || static_cast<std::size_t>(input.nbands) > active_orbitals.size())
    {
        throw std::invalid_argument(
            "FDE embedded_scf requires ks_solver lapack, kpar 1, and nbands within the active AO space");
    }

    const FrozenDensityArtifact active_initial
        = read_density_file(config.active_density_path, config.electron_tolerance);
    if (active_initial.fragment_label != config.active_fragment
        || active_initial.state_label != config.active_state
        || active_initial.alpha_electrons != population.alpha
        || active_initial.beta_electrons != population.beta)
    {
        throw std::invalid_argument(
            "FDE ACTIVE_DENSITY metadata does not match the active state assignment");
    }

    if (config.frozen_density_artifacts.size() + 1 != config.fragments.size())
    {
        throw std::invalid_argument(
            "FDE embedded_scf requires exactly one frozen-density artifact per environment fragment");
    }
    std::vector<FrozenDensityArtifact> frozen_environment;
    std::vector<std::string> seen_labels;
    for (std::size_t index = 0; index < config.frozen_density_artifacts.size(); ++index)
    {
        const RuntimeArtifactPath& path = config.frozen_density_artifacts[index];
        if (path.label == config.active_fragment
            || std::find(seen_labels.begin(), seen_labels.end(), path.label) != seen_labels.end())
        {
            throw std::invalid_argument("FDE frozen environment labels are incomplete or duplicated");
        }
        const FrozenDensityArtifact artifact
            = read_density_file(path.path, config.electron_tolerance);
        if (artifact.fragment_label != path.label || artifact.state_label != config.active_state)
        {
            throw std::invalid_argument(
                "FDE frozen-density metadata does not match FDE_CONFIG");
        }
        frozen_environment.push_back(artifact);
        seen_labels.push_back(path.label);
    }
    std::vector<FrozenDensityArtifact> compatible = frozen_environment;
    compatible.push_back(active_initial);
    DensityArtifactIO::validate_compatible_set(compatible, config.electron_tolerance);

    return std::unique_ptr<FdeLcaoDriver>(
        new FdeLcaoDriver(config,
                          active_orbitals,
                          full_dimension,
                          input.ks_solver,
                          input.kpar,
                          input.nbands,
                          input.nelec,
                          population.alpha,
                          population.beta,
                          active_initial,
                          frozen_environment));
}

const FdeRuntimeConfig& FdeLcaoDriver::config() const
{
    return config_;
}

const std::vector<std::size_t>& FdeLcaoDriver::active_orbitals() const
{
    return active_orbitals_;
}

int FdeLcaoDriver::active_alpha_electrons() const
{
    return active_alpha_electrons_;
}

int FdeLcaoDriver::active_beta_electrons() const
{
    return active_beta_electrons_;
}

void FdeLcaoDriver::initialize_active_charge(Charge& charge) const
{
    if (charge.nspin != 2 || charge.rho == nullptr
        || active_initial_.rho_alpha_bohr3.size()
               != static_cast<std::size_t>(charge.rhopw->nrxx))
    {
        throw std::invalid_argument("FDE active density does not match the ABACUS Charge grid");
    }
    std::copy(active_initial_.rho_alpha_bohr3.begin(),
              active_initial_.rho_alpha_bohr3.end(),
              charge.rho[0]);
    std::copy(active_initial_.rho_beta_bohr3.begin(),
              active_initial_.rho_beta_bohr3.end(),
              charge.rho[1]);
}

void FdeLcaoDriver::attach_embedding_potential(ModulePW::PW_Basis& density_basis,
                                               const UnitCell& unit_cell,
                                               elecstate::Potential& potential)
{
    if (!LibxcPbeProvider::available())
    {
        throw std::runtime_error("FDE embedded_scf requires an ABACUS build with Libxc");
    }
    const FrozenDensityArtifact& reference = frozen_environment_.front();
    SpinDensity frozen;
    frozen.alpha_bohr3.assign(reference.rho_alpha_bohr3.size(), 0.0);
    frozen.beta_bohr3.assign(reference.rho_beta_bohr3.size(), 0.0);
    for (std::size_t fragment = 0; fragment < frozen_environment_.size(); ++fragment)
    {
        for (std::size_t point = 0; point < frozen.alpha_bohr3.size(); ++point)
        {
            frozen.alpha_bohr3[point]
                += frozen_environment_[fragment].rho_alpha_bohr3[point];
            frozen.beta_bohr3[point]
                += frozen_environment_[fragment].rho_beta_bohr3[point];
        }
    }
    const UniformGrid grid = make_grid(reference, density_basis, unit_cell);
    double* density[2] = {frozen.alpha_bohr3.data(), frozen.beta_bohr3.data()};
    const double* const_density[2] = {density[0], density[1]};
    const ModuleBase::matrix frozen_hartree
        = elecstate::H_Hartree_pw::v_hartree(unit_cell,
                                             &density_basis,
                                             2,
                                             const_density);
    std::vector<double> hartree(static_cast<std::size_t>(density_basis.nrxx), 0.0);
    for (int point = 0; point < density_basis.nrxx; ++point)
    {
        hartree[point] = frozen_hartree(0, point);
    }
    PotFdeConfig potential_config;
    potential_config.grid = grid;
    potential_config.kinetic_functional = config_.kinetic_functional;
    potential_config.density_floor_bohr3 = config_.density_floor_bohr3;
    std::shared_ptr<const NonadditiveXcProvider> xc_provider(new LibxcPbeProvider());
    potential.append_component(std::unique_ptr<elecstate::PotBase>(
        new PotFde(&density_basis,
                   frozen,
                   hartree,
                   potential_config,
                   xc_provider)));
}

std::unique_ptr<FdeProjectedHamiltonian> FdeLcaoDriver::projected_hamiltonian(
    hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian) const
{
    return std::unique_ptr<FdeProjectedHamiltonian>(
        new FdeProjectedHamiltonian(full_hamiltonian,
                                    full_ao_dimension_,
                                    active_orbitals_,
                                    1.0e6));
}

void FdeLcaoDriver::solve_projected(
    hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian,
    psi::Psi<double, base_device::DEVICE_CPU>& wavefunctions,
    elecstate::ElecState& electronic_state,
    elecstate::DensityMatrix<double, double>& density_matrix,
    Charge& charge,
    const Parallel_Orbitals& orbitals) const
{
    std::unique_ptr<FdeProjectedHamiltonian> projected
        = this->projected_hamiltonian(full_hamiltonian);
    hsolver::HSolverLCAO<double> solver(&orbitals,
                                       ks_solver_,
                                       kpar_,
                                       static_cast<int>(full_ao_dimension_),
                                       nbands_,
                                       electron_count_,
                                       false);
    solver.solve(projected.get(),
                 wavefunctions,
                 &electronic_state,
                 density_matrix,
                 charge,
                 2,
                 false);
}

} // namespace fde
