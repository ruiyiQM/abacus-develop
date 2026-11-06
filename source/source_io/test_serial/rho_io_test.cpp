#include "source_io/module_output/cube_io.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source_base/global_variable.h"
#include "source_io/module_output/cube_io.h"
#include "prepare_unitcell.h"
#include "source_base/parallel_grid.h"

#include <cstdio>
#include <iterator>
#include <string>

Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
}


Magnetism::~Magnetism()
{
}


#include "source_io/module_parameter/parameter.h"

/***************************************************************
 *  unit test of read_rho, write_rho and trilinear_interpolate
 ***************************************************************/

/**
 * - Tested Functions:
 *   - read_rho()
 *     - the function to read_rho from file
 *     - the serial version without MPI
 *   - trilinear_interpolate()
 *     - the trilinear interpolation method
 *     - the serial version without MPI
 */

class RhoIOTest : public ::testing::Test
{
  protected:
    int nspin = 1;
    int nrxx = 36 * 36 * 36;
    int prenspin = 1;
    double** rho;
    UnitCell* ucell;

    int my_rank = 0;
    std::ofstream ofs_running = std::ofstream("unittest.log");

    void SetUp()
    {
        rho = new double*[nspin];
        ucell = new UnitCell;
        for (int is = 0; is < nspin; ++is)
        {
            rho[is] = new double[nrxx];
        }
    }
    void TearDown()
    {
        for (int is = 0; is < nspin; ++is)
        {
            delete[] rho[is];
        }
        delete[] rho;
        delete ucell;
    }
};

TEST_F(RhoIOTest, Read)
{
    int is = 0;
    std::string fn = "./support/chg.cube";
    int nx = 36;
    int ny = 36;
    int nz = 36;
    double ef;
    UcellTestPrepare utp = UcellTestLib["Si"];
    ucell = utp.SetUcellInfo();
    Parallel_Grid pgrid(nx, ny, nz, nz, nrxx, nz, 1);
    ModuleIO::read_vdata_palgrid(pgrid, my_rank, ofs_running, fn, rho[is], ucell->nat);
    EXPECT_DOUBLE_EQ(rho[0][0], 1.27020863940e-03);
    EXPECT_DOUBLE_EQ(rho[0][46655], 1.33581335706e-02);
}

TEST_F(RhoIOTest, Write)
{
    int nx = 36;
    int ny = 36;
    int nz = 36;
    UcellTestPrepare utp = UcellTestLib["Si"];
    ucell = utp.SetUcellInfo();
    ucell->lat0 = 10.2;
    ucell->latvec = { -0.5,0,0.5,0,0.5,0.5,-0.5,0.5,0 };
    ucell->atoms[0].tau[0] = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    ucell->atoms[0].tau[1] = ModuleBase::Vector3<double>(-0.75, 0.75, 0.75);
    ucell->atoms[0].ncpp.zv = 4;
    Parallel_Grid pgrid(nx, ny, nz, nz, nrxx, nz, 1);
    ModuleIO::read_vdata_palgrid(pgrid, my_rank, ofs_running, "support/chg.cube", rho[0], ucell->nat);
    const std::string path = testing::TempDir() + "abacus_write_vdata.cube";
    ModuleIO::write_vdata_palgrid(pgrid, rho[0], 0, nspin, 0, path, 0.461002, ucell, 11, 1, false, false, false, ofs_running, "charge density");
    std::ifstream written(path);
    std::string line;
    const int header_lines = 6 + ucell->nat;
    for (int i = 0; i < header_lines; ++i)
    {
        std::getline(written, line);
    }
    for (int i = 0; i < nrxx; ++i)
    {
        double value = 0.0;
        written >> value;
        const bool valid = !written.fail();
        ASSERT_TRUE(valid);
        EXPECT_DOUBLE_EQ(value, rho[0][i]);
    }
    written.close();
    std::remove(path.c_str());
}

TEST_F(RhoIOTest, TrilinearInterpolate)
{
    int nx = 36;
    int ny = 40;
    int nz = 44;
    int nx_read = 36;
    int ny_read = 36;
    int nz_read = 36;
    std::ifstream ifs("./support/chg.cube");
    for (int i = 0; i < 8; ++i)
    {
        ifs.ignore(300, '\n');
    }
    std::vector<double> data_read(nx_read * ny_read * nz_read);
    for (int ix = 0; ix < nx_read; ix++)
    {
        for (int iy = 0; iy < ny_read; iy++)
        {
            for (int iz = 0; iz < nz_read; iz++)
            {
                ifs >> data_read[(ix * ny_read + iy) * nz_read + iz];
            }
        }
    }

    // The old implementation is inconsistent: ifdef MPI, [x][y][z]; else, [z][x][y].
    // Now we use [x][y][z] for both MPI and non-MPI, so here we need to chage the index order.
    auto permute_xyz2zxy = [&](const double* const xyz, double* const zxy) -> void
        {
            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    for (int iz = 0; iz < nz; iz++)
                    {
                        zxy[(iz * nx + ix) * ny + iy] = xyz[(ix * ny + iy) * nz + iz];
                    }
                }
            }
        };
    const int nxyz = nx * ny * nz;
    std::vector<double> data_xyz(nxyz);
    std::vector<double> data(nxyz); // z > x > y
    ModuleIO::trilinear_interpolate(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, data_xyz.data());
    permute_xyz2zxy(data_xyz.data(), data.data());
    EXPECT_DOUBLE_EQ(data[0], 0.0010824725010374092);
    EXPECT_DOUBLE_EQ(data[10], 0.058649850374240906);
    EXPECT_DOUBLE_EQ(data[100], 0.018931708073604996);
}


struct CubeIOTest : public ::testing::Test
{
    std::vector<std::string> comment;
    int natom = 0;
    std::vector<double> origin;
    std::vector<int> nvoxel;
    int nx_read = 0;
    int ny_read = 0;
    int nz_read = 0;
    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<double> dz;
    std::vector<std::vector<double>> axis_vecs;
    std::vector<int> atom_type;
    std::vector<double> atom_charge;
    std::vector<std::vector<double>> atom_pos;
    std::vector<double> data_read;
    const std::string fn = "./support/chg.cube";
};


TEST_F(CubeIOTest, ReadCube)
{
    ModuleIO::read_cube(fn, comment, natom, origin, 
     nx_read, ny_read, nz_read, 
     dx, dy, dz, 
     atom_type, atom_charge, atom_pos, data_read);
    EXPECT_EQ(comment[0], "Ionic_Step 1  Cubefile created from ABACUS. Inner loop is z, followed by y and x");
    EXPECT_EQ(comment[1], "1 # number of spin directions 0.461002 # Fermi energy, in Ry");
    EXPECT_EQ(natom, 2);
    for (auto& o : origin) { EXPECT_EQ(o, 0.0); }
    EXPECT_EQ(nx_read, 36);
    EXPECT_EQ(ny_read, 36);
    EXPECT_EQ(nz_read, 36);
    EXPECT_DOUBLE_EQ(dx[0], -0.141667);
    EXPECT_DOUBLE_EQ(dy[2], 0.141667);
    EXPECT_DOUBLE_EQ(dz[1], 0.141667);
    EXPECT_EQ(atom_type.size(), natom);
    EXPECT_EQ(atom_charge.size(), natom);
    EXPECT_EQ(atom_pos.size(), natom);
    for (auto& t : atom_type) { EXPECT_EQ(t, 14); }
    for (auto& c : atom_charge) { EXPECT_DOUBLE_EQ(c, 4.0); }
    EXPECT_DOUBLE_EQ(atom_pos[1][1], 7.65);
    const int nxyz = nx_read * ny_read * nz_read;
    EXPECT_EQ(data_read.size(), nxyz);
    EXPECT_EQ(data_read[1], 2.64004483879e-03);
    EXPECT_EQ(data_read[nxyz - 1], 1.33581335706e-02);
}


TEST_F(CubeIOTest, WriteCube)
{
	ModuleIO::read_cube(fn, comment, natom, origin, 
			nx_read, ny_read, nz_read, 
			dx, dy, dz, 
			atom_type, atom_charge, atom_pos, data_read);

    const std::string path = testing::TempDir() + "abacus_write_cube.cube";
    const std::vector<double> original_data = data_read;
	ModuleIO::write_cube(path,
			comment, natom, origin, 
			nx_read, ny_read, nz_read, 
			dx, dy, dz, atom_type, 
			atom_charge, atom_pos, data_read, 11, false);
    const bool read = ModuleIO::read_cube(path, comment, natom, origin,
                                          nx_read, ny_read, nz_read, dx, dy, dz,
                                          atom_type, atom_charge, atom_pos, data_read);
    ASSERT_TRUE(read);
    EXPECT_EQ(data_read, original_data);
    std::remove(path.c_str());
}

TEST_F(CubeIOTest, PreservesGeometryAtLowDataPrecision)
{
    const std::string path = testing::TempDir() + "abacus_cube_geometry.cube";
    const std::vector<std::string> comments = {"geometry round trip", "low data precision"};
    const double side = 24.0 * 1.8897261254578281;
    const int nx = 192;
    const int ny = 1;
    const int nz = 1;
    const double step = side / nx;
    const std::vector<double> initial_origin = {0.1234567890123456, 0.0, 0.0};
    const std::vector<double> initial_dx = {step, 0.0, 0.0};
    const std::vector<double> initial_dy = {0.0, side, 0.0};
    const std::vector<double> initial_dz = {0.0, 0.0, side};
    const std::vector<int> types = {1};
    const std::vector<double> charges = {1.0};
    const std::vector<std::vector<double>> positions = {{0.3333333333333333, 0.0, 0.0}};
    const std::vector<double> values(nx, 1.0);
    const int atom_count = 1;
    const int data_precision = 2;
    const int values_per_line = 6;
    ModuleIO::write_cube(path, comments, atom_count, initial_origin, nx, ny, nz,
                         initial_dx, initial_dy, initial_dz, types, charges,
                         positions, values, data_precision, false, values_per_line);
    const bool read = ModuleIO::read_cube(path, comment, natom, origin,
                                          nx_read, ny_read, nz_read, dx, dy, dz,
                                          atom_type, atom_charge, atom_pos, data_read);
    ASSERT_TRUE(read);
    EXPECT_DOUBLE_EQ(origin[0], initial_origin[0]);
    EXPECT_DOUBLE_EQ(dx[0], step);
    EXPECT_DOUBLE_EQ(dy[1], side);
    EXPECT_DOUBLE_EQ(dz[2], side);
    EXPECT_DOUBLE_EQ(atom_pos[0][0], positions[0][0]);
    const double volume = nx_read * dx[0] * dy[1] * dz[2];
    const double expected_volume = side * side * side;
    const double tolerance = expected_volume * 1.0e-14;
    EXPECT_NEAR(volume, expected_volume, tolerance);
    EXPECT_EQ(data_read, values);
    std::remove(path.c_str());
}

/**
 * - Tested Functions:
 *   - write_cube()
 *     - append == false truncates the file, leaving exactly one frame
 *     - append == true accumulates one complete frame per call, which is how
 *       out_app_flag == 1 writes successive ionic steps into a fixed filename
 */
TEST_F(CubeIOTest, WriteCubeAppendFrames)
{
    ModuleIO::read_cube(fn, comment, natom, origin,
            nx_read, ny_read, nz_read,
            dx, dy, dz,
            atom_type, atom_charge, atom_pos, data_read);

    const std::string fname = std::string(testing::TempDir()) + "abacus_cube_append.cube";

    auto count_frames = [&fname, this]() {
        std::ifstream ifs(fname.c_str());
        const std::string content((std::istreambuf_iterator<char>(ifs)),
                                  std::istreambuf_iterator<char>());
        ifs.close();
        std::size_t frames = 0;
        std::size_t pos = content.find(comment[0]);
        while (pos != std::string::npos)
        {
            ++frames;
            pos = content.find(comment[0], pos + comment[0].size());
        }
        return frames;
    };

    // The first frame of a run truncates stale content from any previous run.
    ModuleIO::write_cube(fname,
            comment, natom, origin,
            nx_read, ny_read, nz_read,
            dx, dy, dz, atom_type,
            atom_charge, atom_pos, data_read, 11, false);
    EXPECT_EQ(count_frames(), 1u);

    // Two appended frames stack a second and a third complete cube block.
    for (int iframe = 0; iframe < 2; ++iframe)
    {
        ModuleIO::write_cube(fname,
                comment, natom, origin,
                nx_read, ny_read, nz_read,
                dx, dy, dz, atom_type,
                atom_charge, atom_pos, data_read, 11, true);
    }
    EXPECT_EQ(count_frames(), 3u);

    // Every frame ends with a newline, so the next frame starts on a fresh line.
    {
        std::ifstream ifs(fname.c_str());
        const std::string content((std::istreambuf_iterator<char>(ifs)),
                                  std::istreambuf_iterator<char>());
        ifs.close();
        EXPECT_EQ(content.back(), '\n');
    }

    // Overwriting once more drops the two accumulated frames.
    ModuleIO::write_cube(fname,
            comment, natom, origin,
            nx_read, ny_read, nz_read,
            dx, dy, dz, atom_type,
            atom_charge, atom_pos, data_read, 11, false);
    EXPECT_EQ(count_frames(), 1u);

    std::remove(fname.c_str());
}
