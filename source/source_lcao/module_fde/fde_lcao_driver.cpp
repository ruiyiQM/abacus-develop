#include "fde_lcao_driver.h"

#include "fde_grid_partition.h"
#include "fde_kpoint_band_artifact.h"
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
#include "source_base/timer.h"
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
#include <limits>
#include <memory>
#include <stdexcept>

#ifdef __MPI
#include <mpi.h>
#endif

namespace fde
{

namespace
{

class ScopedFdeTimer
{
  public:
    explicit ScopedFdeTimer(const char* name) : name_(name)
    {
        ModuleBase::timer::start("FdeLcaoDriver", name_);
    }

    ~ScopedFdeTimer()
    {
        ModuleBase::timer::end("FdeLcaoDriver", name_);
    }

  private:
    const char* name_;

    ScopedFdeTimer(const ScopedFdeTimer&);
    ScopedFdeTimer& operator=(const ScopedFdeTimer&);
};

FdeRuntimeConfig read_config_file(const std::string& path)
{
    const ScopedFdeTimer timer("read_config");
    std::ifstream input(path.c_str());
    if (!input)
    {
        throw std::runtime_error("Cannot open FDE_CONFIG file: " + path);
    }
    return FdeRuntimeConfigIO::read(input);
}

FrozenDensityArtifact read_density_file_serial(const std::string& path,
                                               const double electron_tolerance)
{
    const ScopedFdeTimer timer("read_density_root");
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Cannot open FDE density artifact: " + path);
    }
    return DensityArtifactIO::read_runtime(input, electron_tolerance);
}

#ifdef __MPI

void require_mpi_success(const int result, const char* operation)
{
    if (result != MPI_SUCCESS)
    {
        throw std::runtime_error(std::string("FDE MPI failure while ") + operation);
    }
}

void broadcast_size(std::size_t& value,
                    const int rank,
                    const int root,
                    MPI_Comm communicator)
{
    unsigned long long transferred
        = rank == root ? static_cast<unsigned long long>(value) : 0ULL;
    require_mpi_success(MPI_Bcast(&transferred,
                                  1,
                                  MPI_UNSIGNED_LONG_LONG,
                                  root,
                                  communicator),
                        "broadcasting an artifact size");
    if (transferred > static_cast<unsigned long long>(
                          std::numeric_limits<std::size_t>::max()))
    {
        throw std::overflow_error("FDE artifact size exceeds local size_t");
    }
    value = static_cast<std::size_t>(transferred);
}

void broadcast_chunks(void* data,
                      const std::size_t count,
                      MPI_Datatype datatype,
                      const int root,
                      MPI_Comm communicator,
                      const char* operation)
{
    std::size_t offset = 0;
    int datatype_size = 0;
    require_mpi_success(MPI_Type_size(datatype, &datatype_size),
                        "querying an artifact MPI datatype");
    if (datatype_size <= 0)
    {
        throw std::runtime_error("FDE artifact MPI datatype has an invalid size");
    }
    unsigned char* bytes = static_cast<unsigned char*>(data);
    while (offset < count)
    {
        const std::size_t remaining = count - offset;
        const int chunk = static_cast<int>(
            std::min(remaining,
                     static_cast<std::size_t>(std::numeric_limits<int>::max())));
        require_mpi_success(MPI_Bcast(bytes + offset * static_cast<std::size_t>(datatype_size),
                                      chunk,
                                      datatype,
                                      root,
                                      communicator),
                            operation);
        offset += static_cast<std::size_t>(chunk);
    }
}

void broadcast_string(std::string& value,
                      const int rank,
                      const int root,
                      MPI_Comm communicator)
{
    std::size_t size = value.size();
    broadcast_size(size, rank, root, communicator);
    if (rank != root)
    {
        value.resize(size);
    }
    if (size != 0)
    {
        broadcast_chunks(&value[0],
                         size,
                         MPI_CHAR,
                         root,
                         communicator,
                         "broadcasting artifact text");
    }
}

void broadcast_density_artifact_metadata(FrozenDensityArtifact& artifact,
                                         const int rank,
                                         const int root,
                                         MPI_Comm communicator)
{
    int integer_fields[6];
    double floating_fields[4];
    std::size_t grid[3];
    if (rank == root)
    {
        integer_fields[0] = artifact.schema_version;
        integer_fields[1] = artifact.alpha_electrons;
        integer_fields[2] = artifact.beta_electrons;
        integer_fields[3] = artifact.freeze_thaw_cycle;
        integer_fields[4] = artifact.scf_converged ? 1 : 0;
        integer_fields[5] = artifact.scf_iterations;
        floating_fields[0] = artifact.cell_volume_bohr3;
        floating_fields[1] = artifact.scf_density_residual;
        floating_fields[2] = artifact.orbital_kinetic_energy_ry;
        floating_fields[3] = artifact.nonlocal_pseudopotential_energy_ry;
        grid[0] = artifact.grid_x;
        grid[1] = artifact.grid_y;
        grid[2] = artifact.grid_z;
    }
    require_mpi_success(MPI_Bcast(integer_fields,
                                  6,
                                  MPI_INT,
                                  root,
                                  communicator),
                        "broadcasting artifact integer metadata");
    require_mpi_success(MPI_Bcast(floating_fields,
                                  4,
                                  MPI_DOUBLE,
                                  root,
                                  communicator),
                        "broadcasting artifact floating-point metadata");
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        broadcast_size(grid[dimension], rank, root, communicator);
    }
    if (rank != root)
    {
        artifact.schema_version = integer_fields[0];
        artifact.alpha_electrons = integer_fields[1];
        artifact.beta_electrons = integer_fields[2];
        artifact.freeze_thaw_cycle = integer_fields[3];
        artifact.scf_converged = integer_fields[4] != 0;
        artifact.scf_iterations = integer_fields[5];
        artifact.cell_volume_bohr3 = floating_fields[0];
        artifact.scf_density_residual = floating_fields[1];
        artifact.orbital_kinetic_energy_ry = floating_fields[2];
        artifact.nonlocal_pseudopotential_energy_ry = floating_fields[3];
        artifact.grid_x = grid[0];
        artifact.grid_y = grid[1];
        artifact.grid_z = grid[2];
    }
    broadcast_string(artifact.fragment_label, rank, root, communicator);
    broadcast_string(artifact.state_label, rank, root, communicator);
    broadcast_string(artifact.geometry_fingerprint, rank, root, communicator);
    broadcast_string(artifact.grid_fingerprint, rank, root, communicator);
    broadcast_string(artifact.pseudopotential_fingerprint, rank, root, communicator);
    broadcast_string(artifact.orbital_fingerprint, rank, root, communicator);
    broadcast_string(artifact.core_density_fingerprint, rank, root, communicator);
    broadcast_string(artifact.xc_functional, rank, root, communicator);
    broadcast_string(artifact.kinetic_functional, rank, root, communicator);
}

#endif

FrozenDensityArtifact read_density_file(const std::string& path,
                                        const double electron_tolerance,
                                        const Parallel_Orbitals& orbitals)
{
#ifdef __MPI
    MPI_Comm communicator = orbitals.comm();
    if (communicator == MPI_COMM_NULL)
    {
        throw std::runtime_error(
            "FDE density input requires an initialized AO communicator");
    }
    int rank = 0;
    int process_count = 0;
    require_mpi_success(MPI_Comm_rank(communicator, &rank),
                        "querying the artifact-input rank");
    require_mpi_success(MPI_Comm_size(communicator, &process_count),
                        "querying the artifact-input communicator size");
    const int root = 0;
    if (root >= process_count)
    {
        throw std::runtime_error("FDE artifact-input root is outside the communicator");
    }

    FrozenDensityArtifact artifact;
    int success = 1;
    std::string error;
    if (rank == root)
    {
        try
        {
            artifact = read_density_file_serial(path, electron_tolerance);
        }
        catch (const std::exception& exception)
        {
            success = 0;
            error = exception.what();
        }
    }
    const ScopedFdeTimer timer("broadcast_density_meta");
    require_mpi_success(MPI_Bcast(&success, 1, MPI_INT, root, communicator),
                        "broadcasting artifact-input status");
    broadcast_string(error, rank, root, communicator);
    if (success == 0)
    {
        throw std::runtime_error(error);
    }
    broadcast_density_artifact_metadata(artifact, rank, root, communicator);
    return artifact;
#else
    (void)orbitals;
    return read_density_file_serial(path, electron_tolerance);
#endif
}

void validate_artifact_set(const FrozenDensityArtifact& active,
                           const std::vector<FrozenDensityArtifact>& frozen,
                           const double electron_tolerance,
                           const Parallel_Orbitals& orbitals)
{
    const ScopedFdeTimer timer("validate_density_set");
#ifdef __MPI
    MPI_Comm communicator = orbitals.comm();
    if (communicator == MPI_COMM_NULL)
    {
        throw std::runtime_error(
            "FDE density validation requires an initialized AO communicator");
    }
    int rank = 0;
    require_mpi_success(MPI_Comm_rank(communicator, &rank),
                        "querying the artifact-validation rank");
    const int root = 0;
    int success = 1;
    std::string error;
    if (rank == root)
    {
        try
        {
            if (frozen.empty())
            {
                throw std::invalid_argument(
                    "FDE compatible artifact set requires at least two fragments");
            }
            DensityArtifactIO::validate(active, electron_tolerance);
            for (std::size_t index = 0; index < frozen.size(); ++index)
            {
                DensityArtifactIO::validate_compatible_pair(active,
                                                            frozen[index],
                                                            electron_tolerance);
            }
        }
        catch (const std::exception& exception)
        {
            success = 0;
            error = exception.what();
        }
    }
    require_mpi_success(MPI_Bcast(&success, 1, MPI_INT, root, communicator),
                        "broadcasting artifact-validation status");
    broadcast_string(error, rank, root, communicator);
    if (success == 0)
    {
        throw std::runtime_error(error);
    }
#else
    (void)orbitals;
    if (frozen.empty())
    {
        throw std::invalid_argument(
            "FDE compatible artifact set requires at least two fragments");
    }
    DensityArtifactIO::validate(active, electron_tolerance);
    for (std::size_t index = 0; index < frozen.size(); ++index)
    {
        DensityArtifactIO::validate_compatible_pair(active,
                                                    frozen[index],
                                                    electron_tolerance);
    }
#endif
}

void release_density_arrays(FrozenDensityArtifact& artifact)
{
    std::vector<double>().swap(artifact.rho_alpha_bohr3);
    std::vector<double>().swap(artifact.rho_beta_bohr3);
}

const RuntimeStateDefinition& active_state(const FdeRuntimeConfig& config)
{
    return config.states[FdeRuntimeConfigIO::state_index(config, config.active_state)];
}

const FragmentChargeSpin& state_assignment(const FdeRuntimeConfig& config,
                                           const std::string& fragment_label)
{
    const RuntimeStateDefinition& state = active_state(config);
    for (std::size_t index = 0; index < state.state.fragments.size(); ++index)
    {
        if (state.state.fragments[index].fragment_label == fragment_label)
        {
            return state.state.fragments[index];
        }
    }
    throw std::invalid_argument("FDE active state does not assign fragment "
                                + fragment_label);
}

const FragmentChargeSpin& active_assignment(const FdeRuntimeConfig& config)
{
    return state_assignment(config, config.active_fragment);
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
            "FDE semilocal KEDF/PBE grid adapter currently requires an orthogonal molecular cell");
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
    const int nspin,
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
      nspin_(nspin),
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
    if (input.nspin != 1 && input.nspin != 2)
    {
        throw std::invalid_argument("FDE embedded_scf requires nspin 1 or 2");
    }
    if (input.nspin == 1)
    {
        const RuntimeStateDefinition& state = active_state(config);
        for (std::size_t index = 0; index < state.state.fragments.size(); ++index)
        {
            const FragmentChargeSpin& assignment = state.state.fragments[index];
            const FragmentDefinition& assigned_fragment
                = config.fragments[FdeRuntimeConfigIO::fragment_index(
                    config, assignment.fragment_label)];
            const SpinPopulation assigned_population
                = StateDefinition::spin_population(assigned_fragment, assignment);
            if (assigned_population.alpha != assigned_population.beta)
            {
                throw std::invalid_argument(
                    "FDE nspin 1 requires every fragment in the active state "
                    "to have an even electron count and spin 0");
            }
        }
    }
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
        = read_density_file(config.active_density_path,
                            config.electron_tolerance,
                            orbitals);
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
            = read_density_file(path.path,
                                config.electron_tolerance,
                                orbitals);
        const FragmentDefinition& environment_fragment
            = config.fragments[FdeRuntimeConfigIO::fragment_index(config, path.label)];
        const SpinPopulation environment_population
            = StateDefinition::spin_population(environment_fragment,
                                               state_assignment(config, path.label));
        if (artifact.fragment_label != path.label || artifact.state_label != config.active_state)
        {
            throw std::invalid_argument(
                "FDE frozen-density metadata does not match FDE_CONFIG");
        }
        if (artifact.alpha_electrons != environment_population.alpha
            || artifact.beta_electrons != environment_population.beta)
        {
            throw std::invalid_argument(
                "FDE frozen-density populations do not match the active state assignment");
        }
        frozen_environment.push_back(artifact);
        seen_labels.push_back(path.label);
    }
    validate_artifact_set(active_initial,
                          frozen_environment,
                          config.electron_tolerance,
                          orbitals);

    return std::unique_ptr<FdeLcaoDriver>(
        new FdeLcaoDriver(config,
                          active_orbitals,
                          full_dimension,
                          input.ks_solver,
                          input.kpar,
                          input.nbands,
                          input.nspin,
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
    if (charge.nspin != nspin_ || charge.rho == nullptr
        || charge.rhopw == nullptr
        || active_alpha_local_.size() != static_cast<std::size_t>(charge.rhopw->nrxx)
        || active_beta_local_.size() != active_alpha_local_.size())
    {
        throw std::invalid_argument(
            "FDE active density has not been partitioned for the ABACUS Charge grid");
    }
    if (nspin_ == 1)
    {
        for (std::size_t point = 0; point < active_alpha_local_.size(); ++point)
        {
            charge.rho[0][point]
                = active_alpha_local_[point] + active_beta_local_[point];
        }
    }
    else
    {
        std::copy(active_alpha_local_.begin(),
                  active_alpha_local_.end(),
                  charge.rho[0]);
        std::copy(active_beta_local_.begin(),
                  active_beta_local_.end(),
                  charge.rho[1]);
    }
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
    {
        const ScopedFdeTimer timer("scatter_active_density");
        active_alpha_local_
            = DensityGridPartition::scatter_from_root(active_initial_.rho_alpha_bohr3,
                                                      density_basis);
        active_beta_local_
            = DensityGridPartition::scatter_from_root(active_initial_.rho_beta_bohr3,
                                                      density_basis);
        if (nspin_ == 1)
        {
            for (std::size_t point = 0; point < active_alpha_local_.size(); ++point)
            {
                const double half_total
                    = 0.5 * (active_alpha_local_[point] + active_beta_local_[point]);
                active_alpha_local_[point] = half_total;
                active_beta_local_[point] = half_total;
            }
        }
        release_density_arrays(active_initial_);
    }
    SpinDensity frozen;
    {
        const ScopedFdeTimer timer("scatter_frozen_density");
        frozen.alpha_bohr3.assign(partition.local_size(), 0.0);
        frozen.beta_bohr3.assign(partition.local_size(), 0.0);
        for (std::size_t fragment = 0; fragment < frozen_environment_.size(); ++fragment)
        {
            const DensityGridPartition fragment_partition
                = DensityGridPartition::from_pw_basis(frozen_environment_[fragment],
                                                      density_basis);
            const std::vector<double> alpha
                = DensityGridPartition::scatter_from_root(
                    frozen_environment_[fragment].rho_alpha_bohr3,
                    density_basis);
            const std::vector<double> beta
                = DensityGridPartition::scatter_from_root(
                    frozen_environment_[fragment].rho_beta_bohr3,
                    density_basis);
            release_density_arrays(frozen_environment_[fragment]);
            for (std::size_t point = 0; point < frozen.alpha_bohr3.size(); ++point)
            {
                frozen.alpha_bohr3[point] += alpha[point];
                frozen.beta_bohr3[point] += beta[point];
            }
        }
        if (nspin_ == 1)
        {
            for (std::size_t point = 0; point < frozen.alpha_bohr3.size(); ++point)
            {
                const double half_total
                    = 0.5 * (frozen.alpha_bohr3[point] + frozen.beta_bohr3[point]);
                frozen.alpha_bohr3[point] = half_total;
                frozen.beta_bohr3[point] = half_total;
            }
        }
    }
    {
        const ScopedFdeTimer timer("build_embedding_potential");
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
        potential.append_component(
            std::unique_ptr<elecstate::PotBase>(component.release()));
    }
}

std::unique_ptr<FdeProjectedHamiltonian> FdeLcaoDriver::projected_hamiltonian(
    hamilt::Hamilt<double, base_device::DEVICE_CPU>& full_hamiltonian,
    const Parallel_Orbitals& orbitals,
    const std::size_t spin_kpoint_count) const
{
    const std::vector<std::size_t> shared_gamma_overlap(spin_kpoint_count, 0);
    return std::unique_ptr<FdeProjectedHamiltonian>(
        new FdeProjectedHamiltonian(full_hamiltonian,
                                    orbitals,
                                    full_ao_dimension_,
                                    active_orbitals_,
                                    1.0e6,
                                    spin_kpoint_count,
                                    shared_gamma_overlap));
}

std::unique_ptr<FdeProjectedHamiltonianComplex>
FdeLcaoDriver::projected_hamiltonian(
    hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>&
        full_hamiltonian,
    const Parallel_Orbitals& orbitals,
    const std::size_t spin_kpoint_count) const
{
    return std::unique_ptr<FdeProjectedHamiltonianComplex>(
        new FdeProjectedHamiltonianComplex(full_hamiltonian,
                                           orbitals,
                                           full_ao_dimension_,
                                           active_orbitals_,
                                           1.0e6,
                                           spin_kpoint_count));
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
        = this->projected_hamiltonian(
            full_hamiltonian,
            orbitals,
            static_cast<std::size_t>(wavefunctions.get_nk()));
    // Both legacy and native ELPA cache the factorized overlap in static
    // solver state.  That optimization is valid for ABACUS' persistent full
    // AO overlap, but FDE creates a new projected overlap object here on every
    // SCF iteration.  Reusing the cache would treat the fresh S matrix as the
    // previous Cholesky inverse and corrupt multi-atom fragment eigenvectors.
    FdeSolverPolicy::prepare_fresh_overlap(ks_solver_);
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
                 nspin_,
                 false);
    if (density_basis_ == nullptr || charge.rhopw != density_basis_
        || charge.nspin != nspin_)
    {
        throw std::runtime_error(
            "FDE spin-density normalization requires the attached PW grid");
    }
    // ABACUS' general LCAO path normalizes only the total density.  The
    // projected alpha and beta quadratures can have different grid errors, so
    // restore both fixed FDE populations before charge/magnetization mixing.
    // Treat this as a spin-branch guard, not an integration-accuracy target.
    // Rescaling small real-space quadrature errors competes with Broyden and
    // creates an SCF floor, while a wrong UKS branch differs by whole
    // electrons.  Final artifacts are still normalized exactly below.
    if (nspin_ == 2)
    {
        normalize_spin_density(charge.rho[0],
                               charge.rho[1],
                               static_cast<std::size_t>(charge.nrxx),
                               active_alpha_electrons_,
                               active_beta_electrons_,
                               active_initial_.cell_volume_bohr3,
                               5.0e-2,
                               *density_basis_);
    }
}

void FdeLcaoDriver::solve_projected(
    hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>&
        full_hamiltonian,
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& wavefunctions,
    elecstate::ElecState& electronic_state,
    elecstate::DensityMatrix<std::complex<double>, double>& density_matrix,
    Charge& charge,
    const Parallel_Orbitals& orbitals) const
{
    std::unique_ptr<FdeProjectedHamiltonianComplex> projected
        = this->projected_hamiltonian(
            full_hamiltonian,
            orbitals,
            static_cast<std::size_t>(wavefunctions.get_nk()));
    // Complex LCAO solvers handle S(k) independently and reset any ELPA
    // factorization state for every k point.  The projected wrapper therefore
    // keeps an independent overlap buffer for each spin-k index.
    hsolver::HSolverLCAO<std::complex<double>> solver(
        &orbitals,
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
                 nspin_,
                 false);
    if (density_basis_ == nullptr || charge.rhopw != density_basis_
        || charge.nspin != nspin_)
    {
        throw std::runtime_error(
            "FDE spin-density normalization requires the attached PW grid");
    }
    if (nspin_ == 2)
    {
        normalize_spin_density(charge.rho[0],
                               charge.rho[1],
                               static_cast<std::size_t>(charge.nrxx),
                               active_alpha_electrons_,
                               active_beta_electrons_,
                               active_initial_.cell_volume_bohr3,
                               5.0e-2,
                               *density_basis_);
    }
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

FdeKPointBandArtifact make_kpoint_band_artifact(
    const elecstate::ElecState& electronic_state,
    const K_Vectors& kpoints,
    const FrozenDensityArtifact& density,
    const std::string& state_label,
    const std::string& fragment_label,
    const std::size_t ao_dimension,
    const int solved_band_count,
    const int nspin)
{
    const int spin_kpoint_count = kpoints.get_nks();
    if ((nspin != 1 && nspin != 2) || spin_kpoint_count <= 0
        || solved_band_count <= 0
        || static_cast<std::size_t>(solved_band_count) > ao_dimension
        || kpoints.isk.size() < static_cast<std::size_t>(spin_kpoint_count)
        || kpoints.wk.size() < static_cast<std::size_t>(spin_kpoint_count)
        || kpoints.kvec_d.size() < static_cast<std::size_t>(spin_kpoint_count))
    {
        throw std::runtime_error("FDE k-point band output contract is incomplete");
    }
    FdeKPointBandArtifact artifact;
    artifact.schema_version = 1;
    artifact.state_label = state_label;
    artifact.fragment_label = fragment_label;
    artifact.geometry_fingerprint = density.geometry_fingerprint;
    artifact.orbital_fingerprint = density.orbital_fingerprint;
    artifact.ao_dimension = ao_dimension;
    artifact.band_count = static_cast<std::size_t>(solved_band_count);
    std::size_t physical_index[2] = {0, 0};
    for (int kpoint = 0; kpoint < spin_kpoint_count; ++kpoint)
    {
        const int spin = kpoints.isk[kpoint];
        if (spin < 0 || spin > 1)
        {
            throw std::runtime_error("FDE k-point band output has an invalid spin index");
        }
        FdeKPointBand point;
        point.spin = spin;
        point.physical_kpoint = physical_index[spin]++;
        point.kx_direct = kpoints.kvec_d[kpoint].x;
        point.ky_direct = kpoints.kvec_d[kpoint].y;
        point.kz_direct = kpoints.kvec_d[kpoint].z;
        point.weight = kpoints.wk[kpoint];
        point.eigenvalues_ry.resize(artifact.band_count);
        for (int band = 0; band < solved_band_count; ++band)
        {
            point.eigenvalues_ry[static_cast<std::size_t>(band)]
                = electronic_state.ekb(kpoint, band);
        }
        artifact.points.push_back(point);
    }
    if (nspin == 1)
    {
        // The artifact schema is explicitly spin resolved.  Duplicate the
        // RKS spatial spectrum so restricted calculations retain the same
        // paired alpha/beta representation used by determinant artifacts.
        // Some ABACUS paths include spin degeneracy in RKS k-point weights,
        // while Gamma-only paths already expose a unit-normalized mesh. Use
        // the runtime sum instead of assuming either convention.
        const std::size_t spatial_point_count = artifact.points.size();
        double spatial_weight_sum = 0.0;
        for (std::size_t index = 0; index < spatial_point_count; ++index)
        {
            spatial_weight_sum += artifact.points[index].weight;
        }
        if (!std::isfinite(spatial_weight_sum) || spatial_weight_sum <= 0.0)
        {
            throw std::runtime_error(
                "FDE restricted k-point output has invalid spatial weights");
        }
        for (std::size_t index = 0; index < spatial_point_count; ++index)
        {
            if (artifact.points[index].spin != 0)
            {
                throw std::runtime_error(
                    "FDE restricted k-point output requires a spatial spin mesh");
            }
            artifact.points[index].weight /= spatial_weight_sum;
            FdeKPointBand beta = artifact.points[index];
            beta.spin = 1;
            artifact.points.push_back(beta);
        }
    }
    return artifact;
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
        || charge.nspin != nspin_
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
    const double* const primary_checkpoint
        = status.converged ? charge.rho_save[0] : charge.rho[0];
    std::vector<double> local_alpha;
    std::vector<double> local_beta;
    EmbeddingPotentialResult embedding;
    {
        const ScopedFdeTimer timer("evaluate_checkpoint");
        if (nspin_ == 1)
        {
            local_alpha.resize(local_grid_size);
            local_beta.resize(local_grid_size);
            for (std::size_t point = 0; point < local_grid_size; ++point)
            {
                local_alpha[point] = 0.5 * primary_checkpoint[point];
                local_beta[point] = local_alpha[point];
            }
        }
        else
        {
            const double* const beta_checkpoint
                = status.converged ? charge.rho_save[1] : charge.rho[1];
            local_alpha.assign(primary_checkpoint,
                               primary_checkpoint + local_grid_size);
            local_beta.assign(beta_checkpoint,
                              beta_checkpoint + local_grid_size);
        }
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
        embedding = embedding_potential_->evaluate(active_density);
    }
    std::vector<double> global_alpha;
    std::vector<double> global_beta;
    {
        const ScopedFdeTimer timer("gather_checkpoint_density");
        global_alpha
            = DensityGridPartition::gather_to_root(local_alpha.data(),
                                                   *density_basis_);
        global_beta
            = DensityGridPartition::gather_to_root(local_beta.data(),
                                                   *density_basis_);
    }

    int spin_kpoint[2] = {-1, -1};
    std::vector<double> global_wavefunctions[2];
    std::vector<double> gathered_overlap;
    std::vector<double> gathered_hamiltonian[2];
    {
        const ScopedFdeTimer timer("gather_ao_artifacts");
        for (int kpoint = 0; kpoint < kpoints.get_nks(); ++kpoint)
        {
            const int spin = kpoints.isk[kpoint];
            if (spin < 0 || spin > 1 || spin_kpoint[spin] != -1)
            {
                throw std::runtime_error(
                    "FDE output requires exactly one Gamma point per spin");
            }
            spin_kpoint[spin] = kpoint;
        }
        if (spin_kpoint[0] < 0 || (nspin_ == 2 && spin_kpoint[1] < 0)
            || kpoints.get_nks() != nspin_)
        {
            throw std::runtime_error("FDE output has an incomplete Gamma spin layout");
        }
        for (int spin = 0; spin < nspin_; ++spin)
        {
            global_wavefunctions[spin]
                = gather_wavefunctions_to_root(wavefunctions,
                                               spin_kpoint[spin],
                                               full_ao_dimension_,
                                               global_band_count,
                                               orbitals,
                                               rank);
        }
        for (int spin = 0; spin < nspin_; ++spin)
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
        if (nspin_ == 1)
        {
            // Determinant artifacts remain explicitly spin resolved. A
            // doubly occupied RKS spatial orbital therefore appears once in
            // each spin block, as required for det(S_alpha) det(S_beta).
            spin_kpoint[1] = spin_kpoint[0];
            global_wavefunctions[1] = global_wavefunctions[0];
            gathered_hamiltonian[1] = gathered_hamiltonian[0];
        }
    }

    if (rank != 0)
    {
        return;
    }

    FrozenDensityArtifact density;
    const std::string density_path
        = config_.output_prefix
          + (status.converged ? ".fde_density" : ".partial.fde_density");
    {
        const ScopedFdeTimer timer("write_density_checkpoint");
        density = active_initial_;
        density.schema_version = 2;
        density.freeze_thaw_cycle = active_initial_.freeze_thaw_cycle + 1;
        density.scf_converged = status.converged;
        density.scf_iterations = status.iterations;
        density.scf_density_residual = status.density_residual;
        density.kinetic_functional
            = kinetic_functional_name(config_.kinetic_functional);
        density.rho_alpha_bohr3.swap(global_alpha);
        density.rho_beta_bohr3.swap(global_beta);
        std::ofstream density_output(density_path.c_str(), std::ios::binary);
        if (!density_output)
        {
            throw std::runtime_error(
                "Cannot create FDE density artifact: " + density_path);
        }
        DensityArtifactIO::write_binary(density_output, density);
        close_artifact(density_output, density_path, "density");
    }

    // A non-self-consistent Hamiltonian, energy, and occupied subspace are not
    // valid inputs to diabatic postprocessing.  Persist only the normalized
    // density checkpoint until the embedded SCF has genuinely converged.
    if (!status.converged)
    {
        return;
    }

    {
        const ScopedFdeTimer timer("write_kpoint_bands");
        const FdeKPointBandArtifact bands
            = make_kpoint_band_artifact(electronic_state,
                                        kpoints,
                                        density,
                                        config_.active_state,
                                        config_.active_fragment,
                                        full_ao_dimension_,
                                        global_band_count,
                                        nspin_);
        FdeKPointBandArtifactIO::validate(
            bands,
            std::max(config_.symmetry_tolerance, 1.0e-12));
        const std::string band_path = config_.output_prefix + ".fde_kbands";
        std::ofstream band_output(band_path.c_str());
        if (!band_output)
        {
            throw std::runtime_error(
                "Cannot create FDE k-point band artifact: " + band_path);
        }
        FdeKPointBandArtifactIO::write(
            band_output,
            bands,
            std::max(config_.symmetry_tolerance, 1.0e-12));
        close_artifact(band_output, band_path, "k-point band");
    }

    {
        const ScopedFdeTimer timer("write_fragment_artifact");
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

        const std::string fragment_path
            = config_.output_prefix + ".fde_fragment";
        std::ofstream fragment_output(fragment_path.c_str());
        if (!fragment_output)
        {
            throw std::runtime_error(
                "Cannot create FDE fragment SCF artifact: " + fragment_path);
        }
        FragmentScfArtifactIO::write(fragment_output,
                                     fragment,
                                     config_.symmetry_tolerance);
        close_artifact(fragment_output, fragment_path, "fragment SCF");
    }
}

void FdeLcaoDriver::write_scf_artifacts(
    Charge& charge,
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& wavefunctions,
    elecstate::ElecState& electronic_state,
    hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>&
        full_hamiltonian,
    const K_Vectors& kpoints,
    const Parallel_Orbitals& orbitals,
    const FdeScfStatus& status)
{
    (void)full_hamiltonian;
    if (embedding_potential_ == nullptr || density_basis_ == nullptr
        || charge.nspin != nspin_ || charge.rhopw == nullptr
        || charge.rho == nullptr || charge.rho_save == nullptr
        || charge.rhopw != density_basis_
        || wavefunctions.get_nk() != kpoints.get_nks()
        || orbitals.get_global_row_size() != static_cast<int>(full_ao_dimension_)
        || orbitals.get_global_col_size() != static_cast<int>(full_ao_dimension_))
    {
        throw std::runtime_error("FDE k-point SCF artifact output contract is incomplete");
    }
    if (status.iterations <= 0 || !std::isfinite(status.density_residual)
        || status.density_residual < 0.0)
    {
        throw std::runtime_error("FDE k-point SCF artifact status is invalid");
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
    const double* const primary_checkpoint
        = status.converged ? charge.rho_save[0] : charge.rho[0];
    std::vector<double> local_alpha;
    std::vector<double> local_beta;
    {
        const ScopedFdeTimer timer("evaluate_checkpoint");
        if (nspin_ == 1)
        {
            local_alpha.resize(local_grid_size);
            local_beta.resize(local_grid_size);
            for (std::size_t point = 0; point < local_grid_size; ++point)
            {
                local_alpha[point] = 0.5 * primary_checkpoint[point];
                local_beta[point] = local_alpha[point];
            }
        }
        else
        {
            const double* const beta_checkpoint
                = status.converged ? charge.rho_save[1] : charge.rho[1];
            local_alpha.assign(primary_checkpoint,
                               primary_checkpoint + local_grid_size);
            local_beta.assign(beta_checkpoint,
                              beta_checkpoint + local_grid_size);
        }
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
        (void)embedding_potential_->evaluate(active_density);
    }
    std::vector<double> global_alpha;
    std::vector<double> global_beta;
    {
        const ScopedFdeTimer timer("gather_checkpoint_density");
        global_alpha = DensityGridPartition::gather_to_root(
            local_alpha.data(), *density_basis_);
        global_beta = DensityGridPartition::gather_to_root(
            local_beta.data(), *density_basis_);
    }
    if (rank != 0)
    {
        return;
    }

    FrozenDensityArtifact density;
    const std::string density_path
        = config_.output_prefix
          + (status.converged ? ".fde_density" : ".partial.fde_density");
    {
        const ScopedFdeTimer timer("write_density_checkpoint");
        density = active_initial_;
        density.schema_version = 2;
        density.freeze_thaw_cycle = active_initial_.freeze_thaw_cycle + 1;
        density.scf_converged = status.converged;
        density.scf_iterations = status.iterations;
        density.scf_density_residual = status.density_residual;
        density.rho_alpha_bohr3.swap(global_alpha);
        density.rho_beta_bohr3.swap(global_beta);
        std::ofstream density_output(density_path.c_str(), std::ios::binary);
        if (!density_output)
        {
            throw std::runtime_error(
                "Cannot create FDE density artifact: " + density_path);
        }
        DensityArtifactIO::write_binary(density_output, density);
        close_artifact(density_output, density_path, "density");
    }
    if (!status.converged)
    {
        return;
    }

    const ScopedFdeTimer timer("write_kpoint_bands");
    const FdeKPointBandArtifact bands
        = make_kpoint_band_artifact(electronic_state,
                                    kpoints,
                                    density,
                                    config_.active_state,
                                    config_.active_fragment,
                                    full_ao_dimension_,
                                    global_band_count,
                                    nspin_);
    FdeKPointBandArtifactIO::validate(
        bands,
        std::max(config_.symmetry_tolerance, 1.0e-12));
    const std::string band_path = config_.output_prefix + ".fde_kbands";
    std::ofstream band_output(band_path.c_str());
    if (!band_output)
    {
        throw std::runtime_error(
            "Cannot create FDE k-point band artifact: " + band_path);
    }
    FdeKPointBandArtifactIO::write(
        band_output,
        bands,
        std::max(config_.symmetry_tolerance, 1.0e-12));
    close_artifact(band_output, band_path, "k-point band");
}

} // namespace fde
