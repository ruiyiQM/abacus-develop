#include "../fde_kpoint_band_artifact.h"

#include <gtest/gtest.h>

#include <sstream>

namespace
{

fde::FdeKPointBandArtifact artifact()
{
    fde::FdeKPointBandArtifact result;
    result.schema_version = 1;
    result.state_label = "neutral";
    result.fragment_label = "Li";
    result.geometry_fingerprint = "lih-primitive";
    result.orbital_fingerprint = "lih-dzvp";
    result.ao_dimension = 4;
    result.band_count = 2;
    for (int spin = 0; spin < 2; ++spin)
    {
        for (std::size_t kpoint = 0; kpoint < 2; ++kpoint)
        {
            fde::FdeKPointBand point;
            point.spin = spin;
            point.physical_kpoint = kpoint;
            point.kx_direct = 0.5 * static_cast<double>(kpoint);
            point.ky_direct = 0.0;
            point.kz_direct = 0.0;
            point.weight = 0.5;
            point.eigenvalues_ry = {
                -0.5 + 0.1 * static_cast<double>(kpoint) + 0.01 * spin,
                0.2 + 0.1 * static_cast<double>(kpoint) + 0.01 * spin,
            };
            result.points.push_back(point);
        }
    }
    return result;
}

} // namespace

TEST(FdeKPointBandArtifact, RoundTripsPairedSpinMesh)
{
    const fde::FdeKPointBandArtifact original = artifact();
    std::ostringstream output;
    fde::FdeKPointBandArtifactIO::write(output, original, 1.0e-12);
    std::istringstream input(output.str());
    const fde::FdeKPointBandArtifact restored
        = fde::FdeKPointBandArtifactIO::read(input, 1.0e-12);

    ASSERT_EQ(restored.points.size(), 4u);
    EXPECT_EQ(restored.points[2].spin, 1);
    EXPECT_EQ(restored.points[3].physical_kpoint, 1u);
    EXPECT_DOUBLE_EQ(restored.points[1].kx_direct, 0.5);
    EXPECT_DOUBLE_EQ(restored.points[3].eigenvalues_ry[1], 0.31);
}

TEST(FdeKPointBandArtifact, RejectsMismatchedSpinMeshes)
{
    fde::FdeKPointBandArtifact invalid = artifact();
    invalid.points[3].kx_direct = 0.25;
    EXPECT_THROW(
        fde::FdeKPointBandArtifactIO::validate(invalid, 1.0e-12),
        std::invalid_argument);
}

TEST(FdeKPointBandArtifact, RejectsUnorderedBandsAndWeights)
{
    fde::FdeKPointBandArtifact invalid = artifact();
    invalid.points[0].eigenvalues_ry = {0.5, -0.5};
    EXPECT_THROW(
        fde::FdeKPointBandArtifactIO::validate(invalid, 1.0e-12),
        std::invalid_argument);

    invalid = artifact();
    invalid.points[0].weight = 0.4;
    EXPECT_THROW(
        fde::FdeKPointBandArtifactIO::validate(invalid, 1.0e-12),
        std::invalid_argument);
}
