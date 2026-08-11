#include "../fde_electronic_coupling.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

fde::OccupiedSpinOrbitals occupied(const std::vector<double>& coefficient,
                                   const std::string& fragment)
{
    fde::OccupiedSpinOrbitals result;
    result.coefficients = coefficient;
    result.orbital_energies_ry = {-1.0};
    result.source_fragment_labels = {fragment};
    return result;
}

fde::DiabaticDeterminantArtifact determinant(const std::string& state,
                                             const std::vector<double>& alpha)
{
    fde::DiabaticDeterminantArtifact result;
    result.schema_version = 1;
    result.state_label = state;
    result.geometry_fingerprint = "geometry";
    result.orbital_fingerprint = "basis";
    result.ao_dimension = 2;
    result.alpha = occupied(alpha, "F");
    result.beta = occupied({0.0, 1.0}, "CH3Cl");
    return result;
}

class DirectionalEnergy : public fde::TransitionEnergyProvider
{
  public:
    std::string name() const override
    {
        return "mpi_directional";
    }

    double evaluate_ry(const fde::DiabaticDeterminantArtifact& bra,
                       const fde::DiabaticDeterminantArtifact&,
                       const fde::SpinTransitionDensityMatrix& density,
                       const std::size_t ao_dimension) const override
    {
        const std::vector<double> overlap = {1.0, 0.0, 0.0, 1.0};
        if (std::fabs(fde::ElectronicCoupling::electron_count(
                          density.alpha, overlap, ao_dimension)
                      - 1.0)
                > 1.0e-12
            || std::fabs(fde::ElectronicCoupling::electron_count(
                             density.beta, overlap, ao_dimension)
                         - 1.0)
                   > 1.0e-12)
        {
            throw std::runtime_error("transition density changed electron count");
        }
        return bra.state_label == "reactant" ? 5.0 : 7.0;
    }
};

} // namespace

TEST(FdeElectronicCouplingMpi, TwoRanksProduceIdenticalResults)
{
    int argument_count = 0;
    char** argument_values = NULL;
    ASSERT_EQ(MPI_Init(&argument_count, &argument_values), MPI_SUCCESS);

    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_success = 1;
    double local_values[4] = {0.0, 0.0, 0.0, 0.0};
    try
    {
        const DirectionalEnergy energy;
        const fde::CouplingValidationControls validation{1.0e-12, 1.0e-12};
        const fde::ElectronicCouplingResult result
            = fde::ElectronicCoupling::evaluate_symmetric(
                determinant("reactant", {1.0, 0.0}),
                determinant("product", {0.8, 0.6}),
                {1.0, 0.0, 0.0, 1.0},
                energy,
                validation,
                1.0e-12);
        local_values[0] = result.normalized_overlap;
        local_values[1] = result.forward_transition_energy_ry;
        local_values[2] = result.reverse_transition_energy_ry;
        local_values[3] = result.hamiltonian_coupling_ry;
    }
    catch (...)
    {
        local_success = 0;
    }

    int global_success = 0;
    double global_minimum[4] = {0.0, 0.0, 0.0, 0.0};
    double global_maximum[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_values, global_minimum, 4, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_values, global_maximum, 4, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Finalize();

    EXPECT_EQ(size, 2);
    EXPECT_EQ(global_success, 1) << "rank " << rank << " failed coupling evaluation";
    EXPECT_NEAR(local_values[0], 0.8, 1.0e-12);
    EXPECT_DOUBLE_EQ(local_values[1], 5.0);
    EXPECT_DOUBLE_EQ(local_values[2], 7.0);
    EXPECT_NEAR(local_values[3], 4.8, 1.0e-12);
    for (int index = 0; index < 4; ++index)
    {
        EXPECT_TRUE(std::isfinite(local_values[index]));
        EXPECT_NEAR(global_minimum[index], global_maximum[index], 1.0e-13);
    }
}
