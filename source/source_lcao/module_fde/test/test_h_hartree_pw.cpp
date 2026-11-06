#include "../coupling/fde_transition_energy_components.h"

#include "source_base/matrix3.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_pot/h_hartree_pw.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

struct FinalizeMpi
{
    ~FinalizeMpi()
    {
        Parallel_Global::finalize_mpi();
    }
};

} // namespace

TEST(FdeNativeHartree, MatchesPeriodicModesSpinCountingAndComplexBilinearEnergy)
{
    int argument_count = 0;
    char** argument_values = nullptr;
    int process_count = 1;
    int thread_count = 1;
    int rank = 0;
    Parallel_Global::read_pal_param(argument_count, argument_values, process_count, thread_count, rank);
    int band_processes = 1;
    int band_rank = 0;
    int band_group = 0;
    int pool_processes = 1;
    int pool_rank = 0;
    int pool = 0;
    Parallel_Global::init_pools(process_count, rank, 1, 1,
                               band_processes, band_rank, band_group,
                               pool_processes, pool_rank, pool);
    int diagonal_rank = 0;
    int diagonal_size = 1;
    int diagonal_color = 0;
    int grid_rank = 0;
    int grid_size = 1;
    Parallel_Global::split_diag_world(1, process_count, rank,
                                     diagonal_rank, diagonal_size, diagonal_color);
    Parallel_Global::split_grid_world(1, process_count, rank, grid_rank, grid_size);
    const FinalizeMpi finalize;

    const double pi = 3.1415926535897932384626433832795;
    const double two_pi = 2.0 * pi;
    const double length = 8.0;
    const double amplitude = 0.001;
    const double imaginary_amplitude = 0.0005;
    const double mean_density = 2.0 / 512.0;
    const double response = 128.0 / pi; // Ry Coulomb response 8*pi/G^2, G=2*pi/8.
    const ModuleBase::Matrix3 lattice(1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0);
    UnitCell cell;
    cell.omega = 512.0;
    cell.tpiba = two_pi / length;
    cell.tpiba2 = cell.tpiba * cell.tpiba;
    ModulePW::PW_Basis basis("cpu", "double");
    basis.initmpi(pool_processes, pool_rank, POOL_WORLD);
    basis.initgrids(length, lattice, 8, 10, 12);
    basis.initparameters(false, 1.0e6, 1, true);
    basis.setuptransform();
    basis.collect_local_pw();
    const std::size_t local_size = static_cast<std::size_t>(basis.nrxx);
    const double volume_element = cell.omega / static_cast<double>(basis.nxyz);
    const double expected_real_energy = cell.omega * response * amplitude * amplitude / 4.0;
    std::vector<double> mode(local_size, 0.0);
    std::vector<double> total(local_size, 0.0);
    std::vector<double> imaginary(local_size, 0.0);
    std::vector<double> alpha(local_size, 0.0);
    std::vector<double> beta(local_size, 0.0);
    std::vector<double> real_potential(local_size, 0.0);
    for (int x = 0; x < basis.nx; ++x)
    {
        const double angle = two_pi * x / basis.nx;
        const double cosine = std::cos(angle);
        for (int y = 0; y < basis.ny; ++y)
        {
            for (int z = 0; z < basis.nplane; ++z)
            {
                const std::size_t point = static_cast<std::size_t>((x * basis.ny + y) * basis.nplane + z);
                mode[point] = cosine;
                total[point] = mean_density + amplitude * cosine;
                imaginary[point] = imaginary_amplitude * cosine;
            }
        }
    }

    const double saved_cache = elecstate::H_Hartree_pw::hartree_energy;
    double worst_potential_error = 0.0;
    for (const int spin_channels : {1, 2})
    {
        for (std::size_t point = 0; point < local_size; ++point)
        {
            alpha[point] = total[point];
            beta[point] = 0.0;
            if (spin_channels == 2)
            {
                alpha[point] = 0.6 * total[point];
                beta[point] = 0.4 * total[point];
            }
        }
        const double* channels[2] = {alpha.data(), beta.data()};
        elecstate::H_Hartree_pw::hartree_energy = 17.0;
        double native_energy = 0.0;
        const auto potential = elecstate::H_Hartree_pw::v_hartree(
            cell, &basis, spin_channels, channels, native_energy);
        EXPECT_DOUBLE_EQ(elecstate::H_Hartree_pw::hartree_energy, 17.0);
        EXPECT_NEAR(native_energy, expected_real_energy, 1.0e-10);
        double maximum_error = 0.0;
        for (std::size_t point = 0; point < local_size; ++point)
        {
            real_potential[point] = potential(0, point);
            const double expected_potential = response * amplitude * mode[point];
            for (int spin = 0; spin < spin_channels; ++spin)
            {
                const double error = std::abs(potential(spin, point) - expected_potential);
                maximum_error = std::max(maximum_error, error);
            }
        }
        Parallel_Reduce::reduce_max(maximum_error);
        worst_potential_error = std::max(worst_potential_error, maximum_error);
        EXPECT_LT(maximum_error, 1.0e-10);
        double contracted = fde::local_hartree_energy_ry(total, real_potential, volume_element);
        Parallel_Reduce::reduce_pool(contracted);
        EXPECT_NEAR(contracted, expected_real_energy, 1.0e-10);
        const auto legacy = elecstate::H_Hartree_pw::v_hartree(cell, &basis, spin_channels, channels);
        EXPECT_NEAR(elecstate::H_Hartree_pw::hartree_energy, native_energy, 1.0e-10);
        EXPECT_EQ(legacy.nr, potential.nr);
    }

    const int one_spin = 1;
    const double* imaginary_channel[1] = {imaginary.data()};
    double native_imaginary_energy = 0.0;
    const double cached_real_energy = elecstate::H_Hartree_pw::hartree_energy;
    const auto imaginary_potential = elecstate::H_Hartree_pw::v_hartree(
        cell, &basis, one_spin, imaginary_channel, native_imaginary_energy);
    EXPECT_DOUBLE_EQ(elecstate::H_Hartree_pw::hartree_energy, cached_real_energy);
    const double expected_imaginary_energy
        = cell.omega * response * imaginary_amplitude * imaginary_amplitude / 4.0;
    EXPECT_NEAR(native_imaginary_energy, expected_imaginary_energy, 1.0e-10);
    std::vector<std::complex<double>> complex_density(local_size, 0.0);
    std::vector<std::complex<double>> complex_potential(local_size, 0.0);
    std::complex<double> particle_count = 0.0;
    for (std::size_t point = 0; point < local_size; ++point)
    {
        const std::complex<double> rho(total[point], imaginary[point]);
        const double imaginary_value = imaginary_potential(0, point);
        const std::complex<double> potential(real_potential[point], imaginary_value);
        complex_density[point] = rho;
        complex_potential[point] = potential;
        particle_count += rho * volume_element;
    }
    auto complex_energy = fde::local_hartree_energy_ry(complex_density, complex_potential, volume_element);
    Parallel_Reduce::reduce_pool(complex_energy);
    Parallel_Reduce::reduce_pool(particle_count);
    const double expected_real_part = expected_real_energy - expected_imaginary_energy;
    const double expected_imaginary_part = cell.omega * response * amplitude * imaginary_amplitude / 2.0;
    EXPECT_NEAR(complex_energy.real(), expected_real_part, 1.0e-10);
    EXPECT_NEAR(complex_energy.imag(), expected_imaginary_part, 1.0e-10);
    EXPECT_NEAR(particle_count.real(), 2.0, 1.0e-12);
    EXPECT_NEAR(particle_count.imag(), 0.0, 1.0e-12);

    std::vector<double> uniform(local_size, mean_density);
    const double* uniform_channel[1] = {uniform.data()};
    double uniform_energy = 0.0;
    const auto uniform_potential = elecstate::H_Hartree_pw::v_hartree(
        cell, &basis, one_spin, uniform_channel, uniform_energy);
    EXPECT_NEAR(uniform_energy, 0.0, 1.0e-12);
    double uniform_maximum = 0.0;
    for (std::size_t point = 0; point < local_size; ++point)
    {
        const double value = std::abs(uniform_potential(0, point));
        uniform_maximum = std::max(uniform_maximum, value);
    }
    Parallel_Reduce::reduce_max(uniform_maximum);
    EXPECT_LT(uniform_maximum, 1.0e-12);
    if (rank == 0)
    {
        std::cout << std::setprecision(17)
                  << "FDE_HARTREE_METRICS {\"ranks\":" << process_count
                  << ",\"real_energy_ry\":" << cached_real_energy
                  << ",\"complex_energy_real_ry\":" << complex_energy.real()
                  << ",\"complex_energy_imaginary_ry\":" << complex_energy.imag()
                  << ",\"particle_count_real\":" << particle_count.real()
                  << ",\"particle_count_imaginary\":" << particle_count.imag()
                  << ",\"maximum_potential_error_ry\":" << worst_potential_error
                  << ",\"uniform_energy_ry\":" << uniform_energy << "}\n";
    }
    elecstate::H_Hartree_pw::hartree_energy = saved_cache;
}
