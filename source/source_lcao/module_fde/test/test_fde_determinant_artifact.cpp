#include "../io/fde_determinant_artifact.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{

using Complex = std::complex<double>;

std::vector<Complex> complex_metric()
{
    return {1.0, Complex(0.0, -0.2), Complex(0.0, 0.2), 1.0};
}

fde::ComplexDiabaticDeterminantArtifact complex_determinant()
{
    fde::ComplexDiabaticDeterminantArtifact value;
    value.schema_version = 2;
    value.state_label = "donor_hole";
    value.geometry_fingerprint = "fixed_cell";
    value.orbital_fingerprint = "basis";
    value.ao_dimension = 2;
    value.boundary_twist = {0.17, 0.11, 0.0};
    value.resource_fingerprint = "fixed_pp_ao_grid_functionals";
    const double coefficient = 1.0 / std::sqrt(1.6);
    value.alpha.coefficients = {coefficient, Complex(0.0, coefficient)};
    value.alpha.orbital_energies_ry = {-1.0};
    value.alpha.source_fragment_labels = {"D"};
    return value;
}

} // namespace

TEST(FdeComplexDeterminantArtifact, UsesConjugateMetricAndPreservesBoundaryAndPhase)
{
    const auto determinant = complex_determinant();
    const auto metric = complex_metric();
    std::ostringstream output;
    fde::ComplexDeterminantArtifactIO::write(output, determinant, metric, 1.0e-12);
    std::istringstream input(output.str());
    const auto restored = fde::ComplexDeterminantArtifactIO::read(input, metric, 1.0e-12);
    EXPECT_EQ(restored.alpha.coefficients, determinant.alpha.coefficients);
    EXPECT_EQ(restored.boundary_twist, determinant.boundary_twist);
    EXPECT_EQ(restored.resource_fingerprint, determinant.resource_fingerprint);
    EXPECT_EQ(restored.beta.coefficients.size(), 0);
    EXPECT_EQ(output.str().find("FDE_DIABATIC_DETERMINANT 2\n"), 0);
}

TEST(FdeComplexDeterminantArtifact, RejectsNonHermitianOrNonfiniteData)
{
    auto determinant = complex_determinant();
    auto metric = complex_metric();
    metric[1] = metric[2];
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::validate(determinant, metric, 1.0e-12),
                 std::invalid_argument);
    metric = complex_metric();
    const double infinity = std::numeric_limits<double>::infinity();
    determinant.alpha.coefficients[0] = Complex(1.0, infinity);
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::validate(determinant, metric, 1.0e-12),
                 std::invalid_argument);
}

TEST(FdeComplexDeterminantArtifact, RequiresCanonicalExplicitBoundaryAndComplexSchema)
{
    auto determinant = complex_determinant();
    const auto metric = complex_metric();
    determinant.boundary_twist.clear();
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::validate(determinant, metric, 1.0e-12),
                 std::invalid_argument);
    determinant.boundary_twist = {0.5, 0.0, 0.0};
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::validate(determinant, metric, 1.0e-12),
                 std::invalid_argument);
    determinant.boundary_twist = {0.0, 0.0, 0.0};
    determinant.schema_version = 1;
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::validate(determinant, metric, 1.0e-12),
                 std::invalid_argument);
}

TEST(FdeComplexDeterminantArtifact, RejectsMalformedComplexRecordAndExcessOccupiedCount)
{
    const auto determinant = complex_determinant();
    const auto metric = complex_metric();
    std::ostringstream output;
    fde::ComplexDeterminantArtifactIO::write(output, determinant, metric, 1.0e-12);
    std::string malformed = output.str();
    const auto coefficient_position = malformed.find('(');
    ASSERT_NE(coefficient_position, std::string::npos);
    malformed.replace(coefficient_position, 1, "broken");
    std::istringstream bad_coefficient(malformed);
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::read(bad_coefficient, metric, 1.0e-12),
                 std::invalid_argument);
    malformed = output.str();
    const auto count_position = malformed.find("ALPHA 1");
    ASSERT_NE(count_position, std::string::npos);
    malformed.replace(count_position, 7, "ALPHA 3");
    std::istringstream bad_count(malformed);
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::read(bad_count, metric, 1.0e-12),
                 std::invalid_argument);
}

TEST(FdeComplexDeterminantArtifact, CannotReadRealSchemaAsComplexOrComplexSchemaAsReal)
{
    const auto determinant = complex_determinant();
    const auto metric = complex_metric();
    std::ostringstream output;
    fde::ComplexDeterminantArtifactIO::write(output, determinant, metric, 1.0e-12);
    std::istringstream complex_input(output.str());
    const std::vector<double> real_metric = {1.0, 0.0, 0.0, 1.0};
    EXPECT_THROW(fde::DeterminantArtifactIO::read(complex_input, real_metric, 1.0e-12),
                 std::invalid_argument);
    std::istringstream real_input("FDE_DIABATIC_DETERMINANT 1\n");
    EXPECT_THROW(fde::ComplexDeterminantArtifactIO::read(real_input, metric, 1.0e-12),
                 std::invalid_argument);
}

TEST(FdeArtifactFile, ReportsAnExistingAbsolutePathAndRejectsAFalseConfirmation)
{
    const std::string path = testing::TempDir() + "fde_determinant_file_notice.data";
    {
        std::ofstream output(path.c_str());
        output << "temporary artifact\n";
    }
    std::ostringstream log;
    EXPECT_NO_THROW(fde::report_artifact_file(path, log, false));
    EXPECT_NE(log.str().find(path), std::string::npos);
    EXPECT_EQ(std::remove(path.c_str()), 0);
    EXPECT_THROW(fde::report_artifact_file(path, log, true), std::runtime_error);
}
