#include "fde_lcao_driver.h"

#include "fde_grid_partition.h"
#include "fde_potential_evaluator.h"
#include "fde_projected_hamiltonian.h"
#include "fde_pw_pool_collectives.h"
#include "fde_solver_policy.h"
#include "fde_spin_density.h"
#include "fde_fragment_artifact.h"
#include "pot_fde.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_cell/unitcell.h"
#include "source_cell/klist.h"
#include "source_estate/elecstate.h"
#include "source_estate/module_charge/charge.h"
#include "source_estate/module_pot/H_Hartree_pw.h"
#include "source_estate/module_pot/potential_new.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_hsolver/hsolver_lcao.h"
#include "source_psi/psi.h"

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
    return DensityArtifactIO::read_runtime(input, electron_tolerance);
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
        || artifact.grid_z != static_cast<std::size_t>(basis.nz))
    {
        throw std::invalid_argument("FDE density artifact does not match the ABACUS grid");
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
      frozen_environment_(frozen_environment),
      density_basis_(nullptr),
      embedding_potential_(nullptr)
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
    if (orbitals.get_global_col_size() != orbitals.get_global_row_size())
    {
        throw std::invalid_argument("FDE embedded_scf requires a square AO distribution");
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
    const bool distributed_ao_matrices
        = orbitals.get_row_size() != orbitals.get_global_row_size()
          || orbitals.get_col_size() != orbitals.get_global_col_size();
    FdeSolverPolicy::validate(input.ks_solver,
                              distributed_ao_matrices,
                              input.kpar);
    if (input.nbands < std::max(population.alpha, population.beta)
        || static_cast<std::size_t>(input.nbands) > active_orbitals.size())
    {
        throw std::invalid_argument(
            "FDE embedded_scf requires nbands within the occupied active AO space");
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
        || charge.rhopw == nullptr
        || active_alpha_local_.size() != static_cast<std::size_t>(charge.rhopw->nrxx)
        || active_beta_local_.size() != active_alpha_local_.size())
    {
        throw std::invalid_argument(
            "FDE active density has not been partitioned for the ABACUS Charge grid");
    }
    std::copy(active_alpha_local_.begin(),
              active_alpha_local_.end(),
              charge.rho[0]);
    std::copy(active_beta_local_.begin(),
              active_beta_local_.end(),
              charge.rho[1]);
}

void FdeLcaoDriver::validate_core_density(const Charge& charge) const
{
    if (charge.rhopw == nullptr || charge.rho_core == nullptr)
    {
        throw std::invalid_argument("FDE core-density validation requires an initialized Charge grid");
    }
    double maximum_core_density = 0.0;
    for (int point = 0; point < charge.rhopw->nrxx; ++point)
    {
        maximum_core_density = std::max(maximum_core_density,
                                        std::fabs(charge.rho_core[point]));
    }
    maximum_core_density
        = PwPoolCollectives::maximum(maximum_core_density, *charge.rhopw);
    if (maximum_core_density > 1.0e-14)
    {
        throw std::invalid_argument(
            "FDE embedded_scf currently requires pseudopotentials without nonlinear core correction");
    }
}

void FdeLcaoDriver::attach_embedding_potential(ModulePW::PW_Basis& density_basis,
                                               const UnitCell& unit_cell,
                                               elecstate::Potential& potential)
{
    density_basis_ = &density_basis;
    if (!LibxcPbeProvider::available())
    {
        throw std::runtime_error("FDE embedded_scf requires an ABACUS build with Libxc");
    }
    if (frozen_environment_.empty())
    {
        throw std::invalid_argument("FDE embedded_scf requires a frozen environment density");
    }
    const FrozenDensityArtifact& reference = frozen_environment_.front();
    const DensityGridPartition partition
        = DensityGridPartition::from_pw_basis(reference, density_basis);
    (void)DensityGridPartition::from_pw_basis(active_initial_, density_basis);
    active_alpha_local_ = partition.extract(active_initial_.rho_alpha_bohr3);
    active_beta_local_ = partition.extract(active_initial_.rho_beta_bohr3);
    SpinDensity frozen;
    frozen.alpha_bohr3.assign(partition.local_size(), 0.0);
    frozen.beta_bohr3.assign(partition.local_size(), 0.0);
    for (std::size_t fragment = 0; fragment < frozen_environment_.size(); ++fragment)
    {
        const DensityGridPartition fragment_partition
            = DensityGridPartition::from_pw_basis(frozen_environment_[fragment], density_basis);
        const std::vector<double> alpha
            = fragment_partition.extract(frozen_environment_[fragment].rho_alpha_bohr3);
        const std::vector<double> beta
            = fragment_partition.extract(frozen_environment_[fragment].rho_beta_bohr3);
        for (std::size_t point = 0; point < frozen.alpha_bohr3.size(); ++point)
        {
            frozen.alpha_bohr3[point] += alpha[point];
            frozen.beta_bohr3[point] += beta[point];
        }
    }
    frozen_alpha_local_ = frozen.alpha_bohr3;
    frozen_beta_local_ = frozen.beta_bohr3;
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
    std::unique_ptr<PotFde> component(new PotFde(&density_basis,
                                                frozen,
                                                hartree,
                                                potential_config,
                                                xc_provider));
    embedding_potential_ = component.get();
    potential.append_component(std::unique_ptr<elecstate::PotBase>(component.release()));
}

std::unique_ptr<FdeProjectedHamiltonian> FdeLcaoDriver::projected_hamiltonian(
    hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian,
    const Parallel_Orbitals& orbitals) const
{
    return std::unique_ptr<FdeProjectedHamiltonian>(
        new FdeProjectedHamiltonian(full_hamiltonian,
                                    orbitals,
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
        = this->projected_hamiltonian(full_hamiltonian, orbitals);
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
    if (density_basis_ == nullptr || charge.rhopw != density_basis_)
    {
        throw std::runtime_error(
            "FDE spin-density normalization requires the attached PW grid");
    }
    // ABACUS' general LCAO path normalizes only the total density.  The
    // projected alpha and beta quadratures can have different grid errors, so
    // restore both fixed FDE populations before charge/magnetization mixing.
    // Ignore sub-microelectron quadrature noise to avoid creating an SCF floor.
    normalize_spin_density(charge.rho[0],
                           charge.rho[1],
                           static_cast<std::size_t>(charge.nrxx),
                           active_alpha_electrons_,
                           active_beta_electrons_,
                           active_initial_.cell_volume_bohr3,
                           1.0e-6,
                           *density_basis_);
}

namespace
{

int ao_rank(const Parallel_Orbitals& orbitals)
{
#ifdef __MPI
    int rank = 0;
    if (orbitals.comm() == MPI_COMM_NULL)
    {
        throw std::runtime_error("FDE artifact output requires an initialized AO communicator");
    }
    MPI_Comm_rank(orbitals.comm(), &rank);
    return rank;
#else
    (void)orbitals;
    return 0;
#endif
}

void validate_parallel_layout(const ModulePW::PW_Basis& basis,
                              const Parallel_Orbitals& orbitals)
{
#ifdef __MPI
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(basis.pool_world, orbitals.comm(), &comparison);
    if ((comparison != MPI_IDENT && comparison != MPI_CONGRUENT)
        || ao_rank(orbitals) != basis.poolrank)
    {
        throw std::runtime_error(
            "FDE artifact output requires congruent PW and AO communicators");
    }
#else
    (void)basis;
    (void)orbitals;
#endif
}

std::vector<double> gather_matrix_to_root(
    const hamilt::MatrixBlock<double>& matrix,
    const std::size_t dimension,
    const Parallel_Orbitals& orbitals,
    const int rank)
{
    if ((matrix.row * matrix.col != 0 && matrix.p == nullptr)
        || matrix.row != static_cast<std::size_t>(orbitals.get_row_size())
        || matrix.col != static_cast<std::size_t>(orbitals.get_col_size()))
    {
        throw std::runtime_error("FDE AO matrix does not match Parallel_Orbitals");
    }
    std::vector<double> output(
        rank == 0 ? dimension * dimension : std::size_t{0});
#ifdef __MPI
    if (matrix.desc == nullptr
        || matrix.desc[2] != static_cast<int>(dimension)
        || matrix.desc[3] != static_cast<int>(dimension))
    {
        throw std::runtime_error("FDE AO matrix has an invalid ScaLAPACK descriptor");
    }
    Parallel_2D global;
    global.set(static_cast<int>(dimension),
               static_cast<int>(dimension),
               static_cast<int>(dimension),
               orbitals.blacs_ctxt);
    Cpxgemr2d(static_cast<int>(dimension),
              static_cast<int>(dimension),
              matrix.p,
              1,
              1,
              const_cast<int*>(matrix.desc),
              output.data(),
              1,
              1,
              global.desc,
              global.blacs_ctxt);
#else
    std::copy(matrix.p, matrix.p + dimension * dimension, output.begin());
#endif
    return output;
}

std::vector<double> gather_wavefunctions_to_root(
    psi::Psi<double, base_device::DEVICE_CPU>& wavefunctions,
    const int kpoint,
    const std::size_t dimension,
    const int global_band_count,
    const Parallel_Orbitals& orbitals,
    const int rank)
{
    if (global_band_count <= 0)
    {
        throw std::runtime_error("FDE wavefunction gather requires solved bands");
    }
    wavefunctions.fix_k(kpoint);
    std::vector<double> output(
        rank == 0 ? dimension * static_cast<std::size_t>(global_band_count)
                  : std::size_t{0});
#ifdef __MPI
    if (orbitals.desc_wfc[2] != static_cast<int>(dimension)
        || orbitals.desc_wfc[3] != global_band_count)
    {
        throw std::runtime_error("FDE wavefunction descriptor has invalid global dimensions");
    }
    Parallel_2D global;
    global.set(static_cast<int>(dimension),
               global_band_count,
               std::max(static_cast<int>(dimension), global_band_count),
               orbitals.blacs_ctxt);
    Cpxgemr2d(static_cast<int>(dimension),
              global_band_count,
              wavefunctions.get_pointer(),
              1,
              1,
              const_cast<int*>(orbitals.desc_wfc),
              output.data(),
              1,
              1,
              global.desc,
              global.blacs_ctxt);
#else
    if (wavefunctions.get_nbasis() != static_cast<int>(dimension)
        || wavefunctions.get_nbands() != global_band_count)
    {
        throw std::runtime_error("FDE serial wavefunction dimensions are invalid");
    }
    for (int band = 0; band < global_band_count; ++band)
    {
        for (std::size_t ao = 0; ao < dimension; ++ao)
        {
            output[ao + static_cast<std::size_t>(band) * dimension]
                = wavefunctions(band, static_cast<int>(ao));
        }
    }
#endif
    return output;
}

void append_occupied_spin(const std::vector<double>& global_wavefunctions,
                          const elecstate::ElecState& electronic_state,
                          const int kpoint,
                          const int electron_count,
                          const std::size_t dimension,
                          const int solved_band_count,
                          const std::string& fragment_label,
                          OccupiedSpinOrbitals& output)
{
    if (electron_count < 0 || electron_count > solved_band_count
        || global_wavefunctions.size()
               != dimension * static_cast<std::size_t>(solved_band_count))
    {
        throw std::runtime_error("FDE occupied population does not fit the solved band space");
    }
    output.coefficients.assign(dimension * static_cast<std::size_t>(electron_count), 0.0);
    output.orbital_energies_ry.resize(static_cast<std::size_t>(electron_count));
    output.source_fragment_labels.assign(static_cast<std::size_t>(electron_count),
                                         fragment_label);
    for (int band = 0; band < electron_count; ++band)
    {
        output.orbital_energies_ry[band] = electronic_state.ekb(kpoint, band);
        for (std::size_t ao = 0; ao < dimension; ++ao)
        {
            output.coefficients[ao + static_cast<std::size_t>(band) * dimension]
                = global_wavefunctions[ao + static_cast<std::size_t>(band) * dimension];
        }
    }
}

void close_artifact(std::ofstream& output,
                    const std::string& path,
                    const char* description)
{
    output.flush();
    if (!output)
    {
        throw std::runtime_error(std::string("Failed to flush FDE ")
                                 + description + " artifact: " + path);
    }
    output.close();
    if (!output)
    {
        throw std::runtime_error(std::string("Failed to close FDE ")
                                 + description + " artifact: " + path);
    }
}

} // namespace

void FdeLcaoDriver::write_scf_artifacts(
    Charge& charge,
    psi::Psi<double, base_device::DEVICE_CPU>& wavefunctions,
    elecstate::ElecState& electronic_state,
    hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian,
    const K_Vectors& kpoints,
    const Parallel_Orbitals& orbitals,
    const FdeScfStatus& status)
{
    if (embedding_potential_ == nullptr || density_basis_ == nullptr
        || charge.nspin != 2
        || charge.rhopw == nullptr || charge.rho == nullptr
        || charge.rho_save == nullptr
        || charge.rhopw != density_basis_
        || wavefunctions.get_nk() != kpoints.get_nks()
        || orbitals.get_global_row_size() != static_cast<int>(full_ao_dimension_)
        || orbitals.get_global_col_size() != static_cast<int>(full_ao_dimension_))
    {
        throw std::runtime_error("FDE SCF artifact output contract is incomplete");
    }
    if (status.iterations <= 0 || !std::isfinite(status.density_residual)
        || status.density_residual < 0.0)
    {
        throw std::runtime_error("FDE SCF artifact status is invalid");
    }
    validate_parallel_layout(*density_basis_, orbitals);
    const int rank = ao_rank(orbitals);
#ifdef __MPI
    const int global_band_count = orbitals.get_wfc_global_nbands();
#else
    const int global_band_count = wavefunctions.get_nbands();
#endif

    const std::size_t local_grid_size
        = static_cast<std::size_t>(density_basis_->nrxx);
    // rho_save is the density that generated the final converged Hamiltonian.
    // At nmax, charge.rho instead contains the newly mixed and renormalized
    // checkpoint that should seed the next inexact freeze--thaw update.
    const double* const alpha_checkpoint
        = status.converged ? charge.rho_save[0] : charge.rho[0];
    const double* const beta_checkpoint
        = status.converged ? charge.rho_save[1] : charge.rho[1];
    std::vector<double> local_alpha(alpha_checkpoint,
                                    alpha_checkpoint + local_grid_size);
    std::vector<double> local_beta(beta_checkpoint,
                                   beta_checkpoint + local_grid_size);
    normalize_nonnegative_spin_density(local_alpha.data(),
                                       local_beta.data(),
                                       local_grid_size,
                                       active_alpha_electrons_,
                                       active_beta_electrons_,
                                       active_initial_.cell_volume_bohr3,
                                       *density_basis_);
    SpinDensity active_density;
    active_density.alpha_bohr3 = local_alpha;
    active_density.beta_bohr3 = local_beta;
    const EmbeddingPotentialResult embedding
        = embedding_potential_->evaluate(active_density);
    std::vector<double> global_alpha
        = DensityGridPartition::gather_to_root(local_alpha.data(), *density_basis_);
    std::vector<double> global_beta
        = DensityGridPartition::gather_to_root(local_beta.data(), *density_basis_);

    int spin_kpoint[2] = {-1, -1};
    for (int kpoint = 0; kpoint < kpoints.get_nks(); ++kpoint)
    {
        const int spin = kpoints.isk[kpoint];
        if (spin < 0 || spin > 1 || spin_kpoint[spin] != -1)
        {
            throw std::runtime_error("FDE output requires exactly one Gamma point per spin");
        }
        spin_kpoint[spin] = kpoint;
    }
    if (spin_kpoint[0] < 0 || spin_kpoint[1] < 0)
    {
        throw std::runtime_error("FDE output is missing a collinear spin channel");
    }
    std::vector<double> global_wavefunctions[2];
    for (int spin = 0; spin < 2; ++spin)
    {
        global_wavefunctions[spin]
            = gather_wavefunctions_to_root(wavefunctions,
                                           spin_kpoint[spin],
                                           full_ao_dimension_,
                                           global_band_count,
                                           orbitals,
                                           rank);
    }

    std::vector<double> gathered_overlap;
    std::vector<double> gathered_hamiltonian[2];
    for (int spin = 0; spin < 2; ++spin)
    {
        full_hamiltonian.updateHk(spin_kpoint[spin]);
        hamilt::MatrixBlock<double> local_hamiltonian;
        hamilt::MatrixBlock<double> local_overlap;
        full_hamiltonian.matrix(local_hamiltonian, local_overlap);
        if (spin == 0)
        {
            gathered_overlap = gather_matrix_to_root(local_overlap,
                                                     full_ao_dimension_,
                                                     orbitals,
                                                     rank);
        }
        gathered_hamiltonian[spin]
            = gather_matrix_to_root(local_hamiltonian,
                                    full_ao_dimension_,
                                    orbitals,
                                    rank);
    }

    if (rank != 0)
    {
        return;
    }

    FrozenDensityArtifact density = active_initial_;
    density.schema_version = 2;
    density.freeze_thaw_cycle = active_initial_.freeze_thaw_cycle + 1;
    density.scf_converged = status.converged;
    density.scf_iterations = status.iterations;
    density.scf_density_residual = status.density_residual;
    density.rho_alpha_bohr3.swap(global_alpha);
    density.rho_beta_bohr3.swap(global_beta);
    const std::string density_path
        = config_.output_prefix
          + (status.converged ? ".fde_density" : ".partial.fde_density");
    std::ofstream density_output(density_path.c_str());
    if (!density_output)
    {
        throw std::runtime_error("Cannot create FDE density artifact: " + density_path);
    }
    DensityArtifactIO::write(density_output, density);
    close_artifact(density_output, density_path, "density");

    // A non-self-consistent Hamiltonian, energy, and occupied subspace are not
    // valid inputs to diabatic postprocessing.  Persist only the normalized
    // density checkpoint until the embedded SCF has genuinely converged.
    if (!status.converged)
    {
        return;
    }

    FragmentScfArtifact fragment;
    fragment.schema_version = 1;
    fragment.state_label = config_.active_state;
    fragment.fragment_label = config_.active_fragment;
    fragment.geometry_fingerprint = density.geometry_fingerprint;
    fragment.orbital_fingerprint = density.orbital_fingerprint;
    fragment.density_path = density_path;
    fragment.freeze_thaw_cycle = density.freeze_thaw_cycle;
    fragment.scf_converged = true;
    fragment.ao_dimension = full_ao_dimension_;
    fragment.active_orbitals = active_orbitals_;
    append_occupied_spin(global_wavefunctions[0],
                         electronic_state,
                         spin_kpoint[0],
                         active_alpha_electrons_,
                         full_ao_dimension_,
                         global_band_count,
                         config_.active_fragment,
                         fragment.alpha);
    append_occupied_spin(global_wavefunctions[1],
                         electronic_state,
                         spin_kpoint[1],
                         active_beta_electrons_,
                         full_ao_dimension_,
                         global_band_count,
                         config_.active_fragment,
                         fragment.beta);
    fragment.ao_overlap.swap(gathered_overlap);
    fragment.hamiltonian_alpha_ry.swap(gathered_hamiltonian[0]);
    fragment.hamiltonian_beta_ry.swap(gathered_hamiltonian[1]);
    fragment.subsystem_total_energy_ry = electronic_state.f_en.etot;
    fragment.ion_ion_energy_ry = electronic_state.f_en.ewald_energy;
    fragment.hartree_cross_energy_ry = embedding.hartree_cross_energy_ry;
    fragment.nonadditive_kinetic_energy_ry
        = embedding.nonadditive_kinetic_energy_ry;
    fragment.nonadditive_xc_energy_ry = embedding.nonadditive_xc_energy_ry;

    const std::string fragment_path = config_.output_prefix + ".fde_fragment";
    std::ofstream fragment_output(fragment_path.c_str());
    if (!fragment_output)
    {
        throw std::runtime_error("Cannot create FDE fragment SCF artifact: " + fragment_path);
    }
    FragmentScfArtifactIO::write(fragment_output,
                                 fragment,
                                 config_.symmetry_tolerance);
    close_artifact(fragment_output, fragment_path, "fragment SCF");
}

} // namespace fde
