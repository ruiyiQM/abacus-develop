#include "../fde_pw_grid_differential.h"

#include "source_base/matrix3.h"
#include "source_basis/module_pw/pw_basis.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{

const double two_pi = 6.283185307179586476925286766559;

void initialize_basis(ModulePW::PW_Basis& basis,
                      const int process_count,
                      const int rank)
{
    basis.initmpi(process_count, rank, MPI_COMM_WORLD);
    const ModuleBase::Matrix3 lattice(1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0);
    basis.initgrids(1.0, lattice, 12, 10, 16);
    basis.initparameters(false, 1.0e6, 1, true);
    basis.setuptransform();
    basis.collect_local_pw();
}

} // namespace

TEST(FdePwGridDifferentialMpi, DifferentiatesPeriodicModesAcrossZSlabs)
{
    int process_count = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ModulePW::PW_Basis basis("cpu", "double");
    initialize_basis(basis, process_count, rank);
    const fde::PwGridDifferential differential(basis, false);

    std::vector<double> scalar(static_cast<std::size_t>(basis.nrxx), 0.0);
    std::vector<double> vector_x(static_cast<std::size_t>(basis.nrxx), 0.0);
    std::vector<double> vector_y(static_cast<std::size_t>(basis.nrxx), 0.0);
    std::vector<double> vector_z(static_cast<std::size_t>(basis.nrxx), 0.0);
    for (int ix = 0; ix < basis.nx; ++ix)
    {
        const double x = static_cast<double>(ix) / static_cast<double>(basis.nx);
        for (int iy = 0; iy < basis.ny; ++iy)
        {
            const double y = static_cast<double>(iy) / static_cast<double>(basis.ny);
            const int xy = ix * basis.ny + iy;
            for (int local_z = 0; local_z < basis.nplane; ++local_z)
            {
                const int global_z = basis.startz_current + local_z;
                const double z
                    = static_cast<double>(global_z) / static_cast<double>(basis.nz);
                const std::size_t index
                    = static_cast<std::size_t>(xy * basis.nplane + local_z);
                scalar[index] = std::sin(two_pi * x)
                                + 2.0 * std::cos(two_pi * y)
                                + 3.0 * std::sin(two_pi * z);
                vector_x[index] = std::sin(two_pi * x);
                vector_y[index] = 2.0 * std::sin(two_pi * y);
                vector_z[index] = 3.0 * std::cos(two_pi * z);
            }
        }
    }

    std::vector<double> gradient_x;
    std::vector<double> gradient_y;
    std::vector<double> gradient_z;
    differential.gradient(scalar, gradient_x, gradient_y, gradient_z);
    const std::vector<double> divergence
        = differential.divergence(vector_x, vector_y, vector_z);

    double local_maximum_error = 0.0;
    for (int ix = 0; ix < basis.nx; ++ix)
    {
        const double x = static_cast<double>(ix) / static_cast<double>(basis.nx);
        for (int iy = 0; iy < basis.ny; ++iy)
        {
            const double y = static_cast<double>(iy) / static_cast<double>(basis.ny);
            const int xy = ix * basis.ny + iy;
            for (int local_z = 0; local_z < basis.nplane; ++local_z)
            {
                const int global_z = basis.startz_current + local_z;
                const double z
                    = static_cast<double>(global_z) / static_cast<double>(basis.nz);
                const std::size_t index
                    = static_cast<std::size_t>(xy * basis.nplane + local_z);
                local_maximum_error = std::max(
                    local_maximum_error,
                    std::fabs(gradient_x[index] - two_pi * std::cos(two_pi * x)));
                local_maximum_error = std::max(
                    local_maximum_error,
                    std::fabs(gradient_y[index] + 2.0 * two_pi * std::sin(two_pi * y)));
                local_maximum_error = std::max(
                    local_maximum_error,
                    std::fabs(gradient_z[index] - 3.0 * two_pi * std::cos(two_pi * z)));
                const double expected_divergence
                    = two_pi * std::cos(two_pi * x)
                      + 2.0 * two_pi * std::cos(two_pi * y)
                      - 3.0 * two_pi * std::sin(two_pi * z);
                local_maximum_error = std::max(
                    local_maximum_error,
                    std::fabs(divergence[index] - expected_divergence));
            }
        }
    }
    double global_maximum_error = 0.0;
    MPI_Allreduce(&local_maximum_error,
                  &global_maximum_error,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_LT(global_maximum_error, 1.0e-10);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
