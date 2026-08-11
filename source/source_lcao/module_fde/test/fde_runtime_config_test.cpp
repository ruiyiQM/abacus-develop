#include "../fde_runtime_config.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

const char* fluoride_substitution_config()
{
    return R"(FDE_CONFIG 1
ATOM_COUNT 6
FRAGMENT F 7 1 0
FRAGMENT CH3Cl 14 5 1 2 3 4 5
STATE reactant -1 0 2 F -1 0 CH3Cl 0 0
STATE product -1 0 2 F 0 1 CH3Cl -1 -1
ACTIVE_STATE reactant
ACTIVE_FRAGMENT F
ACTIVE_DENSITY artifacts/reactant_F.fde_density
FROZEN_DENSITY CH3Cl artifacts/reactant_CH3Cl.fde_density
DETERMINANT reactant artifacts/reactant.fde_determinant
DETERMINANT product artifacts/product.fde_determinant
LINEARIZED_STATE reactant artifacts/reactant.fde_linearized_state
LINEARIZED_STATE product artifacts/product.fde_linearized_state
DIAGONAL_ENERGY_RY reactant -80.1
DIAGONAL_ENERGY_RY product -80.0
AO_OVERLAP artifacts/S_gamma.fde_matrix
OUTPUT_PREFIX artifacts/reactant_F
KEDF pw91k
DENSITY_FLOOR_BOHR3 1e-12
MAX_SCF_ITERATIONS 120
SCF_DENSITY_TOLERANCE 1e-8
ELECTRON_TOLERANCE 1e-8
MIXING_BETA 0.25
MAX_FREEZE_THAW_CYCLES 30
FREEZE_THAW_DENSITY_TOLERANCE 1e-7
ENERGY_TOLERANCE_RY 1e-8
UPDATE_ORDER 2 F CH3Cl
K_STATES 2 reactant product
L_FRAGMENTS 2 F CH3Cl
M_FRAGMENTS 2 F CH3Cl
SINGULAR_VALUE_TOLERANCE 1e-10
OVERLAP_EIGENVALUE_CUTOFF 1e-9
SYMMETRY_TOLERANCE 1e-10
RESIDUAL_TOLERANCE 1e-8
MINIMUM_ROOT_OVERLAP 0.5
CALCULATE_FORCE true
HARTREE_RECIPROCITY_TOLERANCE_RY 1e-8
END_FDE_CONFIG
)";
}

} // namespace

TEST(FdeRuntimeConfig, ParsesFluorideSubstitutionTwoStateModel)
{
    std::istringstream input(fluoride_substitution_config());
    const fde::FdeRuntimeConfig config = fde::FdeRuntimeConfigIO::read(input);

    ASSERT_EQ(config.fragments.size(), 2);
    EXPECT_EQ(config.fragments[0].label, "F");
    EXPECT_EQ(config.fragments[1].neutral_valence_electrons, 14);
    ASSERT_EQ(config.states.size(), 2);
    EXPECT_EQ(config.states[0].state.fragments[0].charge, -1);
    EXPECT_EQ(config.states[1].state.fragments[0].spin_projection, 1);
    EXPECT_EQ(config.states[1].state.fragments[1].spin_projection, -1);
    EXPECT_EQ(config.active_state, "reactant");
    EXPECT_EQ(config.active_fragment, "F");
    EXPECT_EQ(config.active_density_path, "artifacts/reactant_F.fde_density");
    EXPECT_EQ(config.maximum_scf_iterations, 120);
    EXPECT_EQ(config.fragment_xc, "pbe");
    EXPECT_EQ(config.embedding_xc, "pbe");
    EXPECT_EQ(config.coupling_provider, "symmetric_linearized");
    EXPECT_DOUBLE_EQ(config.transition_density_trace_tolerance, 1.0e-8);
    EXPECT_DOUBLE_EQ(config.mixing_beta, 0.25);
    EXPECT_TRUE(config.calculate_force);
    ASSERT_EQ(config.linearized_state_artifacts.size(), 2);
    EXPECT_EQ(config.linearized_state_artifacts[1].label, "product");
    EXPECT_EQ(fde::FdeRuntimeConfigIO::state_index(config, "product"), 1);
}

TEST(FdeRuntimeConfig, ParsesAndCanonicalizesIndependentXcChoices)
{
    std::string text(fluoride_substitution_config());
    const std::string anchor = "KEDF pw91k";
    text.insert(text.find(anchor), "FRAGMENT_XC PBE0\nEMBEDDING_XC PBE\n");
    std::istringstream input(text);
    const fde::FdeRuntimeConfig config = fde::FdeRuntimeConfigIO::read(input);
    EXPECT_EQ(config.fragment_xc, "pbe0");
    EXPECT_EQ(config.embedding_xc, "pbe");

    std::ostringstream serialized;
    fde::FdeRuntimeConfigIO::write(serialized, config);
    EXPECT_NE(serialized.str().find("FRAGMENT_XC pbe0\n"), std::string::npos);
    EXPECT_NE(serialized.str().find("EMBEDDING_XC pbe\n"), std::string::npos);
}

TEST(FdeRuntimeConfig, RejectsNonPbeEmbeddingXc)
{
    std::string text(fluoride_substitution_config());
    text.insert(text.find("KEDF pw91k"), "EMBEDDING_XC pbe0\n");
    std::istringstream input(text);
    EXPECT_THROW(fde::FdeRuntimeConfigIO::read(input), std::invalid_argument);
}

TEST(FdeRuntimeConfig, CanonicalizesCouplingProviderAndTraceTolerance)
{
    std::string text(fluoride_substitution_config());
    text.insert(text.find("KEDF pw91k"),
                "COUPLING_PROVIDER linearized\n"
                "TRANSITION_DENSITY_TRACE_TOLERANCE 2e-9\n");
    std::istringstream input(text);
    const fde::FdeRuntimeConfig config = fde::FdeRuntimeConfigIO::read(input);
    EXPECT_EQ(config.coupling_provider, "symmetric_linearized");
    EXPECT_DOUBLE_EQ(config.transition_density_trace_tolerance, 2.0e-9);
}

TEST(FdeRuntimeConfig, RoundTripsDeterministically)
{
    std::istringstream input(fluoride_substitution_config());
    const fde::FdeRuntimeConfig first = fde::FdeRuntimeConfigIO::read(input);
    std::ostringstream serialized;
    fde::FdeRuntimeConfigIO::write(serialized, first);
    std::istringstream roundtrip(serialized.str());
    const fde::FdeRuntimeConfig second = fde::FdeRuntimeConfigIO::read(roundtrip);

    EXPECT_EQ(second.fragments.size(), first.fragments.size());
    EXPECT_EQ(second.states.size(), first.states.size());
    EXPECT_EQ(second.k_state_labels, first.k_state_labels);
    EXPECT_EQ(second.frozen_density_artifacts[0].path,
              first.frozen_density_artifacts[0].path);
    EXPECT_DOUBLE_EQ(second.diagonal_energies[1].energy_ry,
                     first.diagonal_energies[1].energy_ry);
}

TEST(FdeRuntimeConfig, AcceptsThomasFermiAliasAndWritesCanonicalName)
{
    std::string text(fluoride_substitution_config());
    text.replace(text.find("KEDF pw91k"), std::string("KEDF pw91k").size(), "KEDF tf");
    std::istringstream input(text);
    const fde::FdeRuntimeConfig config = fde::FdeRuntimeConfigIO::read(input);
    EXPECT_EQ(config.kinetic_functional, fde::KineticFunctional::ThomasFermi);

    std::ostringstream serialized;
    fde::FdeRuntimeConfigIO::write(serialized, config);
    EXPECT_NE(serialized.str().find("KEDF thomas_fermi\n"), std::string::npos);
}

TEST(FdeRuntimeConfig, AcceptsLegacyLc94NameAndWritesPw91k)
{
    std::string text(fluoride_substitution_config());
    text.replace(text.find("KEDF pw91k"),
                 std::string("KEDF pw91k").size(),
                 "KEDF lc94");
    std::istringstream input(text);
    const fde::FdeRuntimeConfig config = fde::FdeRuntimeConfigIO::read(input);
    EXPECT_EQ(config.kinetic_functional, fde::KineticFunctional::Pw91k);

    std::ostringstream serialized;
    fde::FdeRuntimeConfigIO::write(serialized, config);
    EXPECT_NE(serialized.str().find("KEDF pw91k\n"), std::string::npos);
}

TEST(FdeRuntimeConfig, RoundTripsRevApbek)
{
    std::string text(fluoride_substitution_config());
    text.replace(text.find("KEDF pw91k"),
                 std::string("KEDF pw91k").size(),
                 "KEDF revapbek");
    std::istringstream input(text);
    const fde::FdeRuntimeConfig config = fde::FdeRuntimeConfigIO::read(input);
    EXPECT_EQ(config.kinetic_functional, fde::KineticFunctional::RevApbek);

    std::ostringstream serialized;
    fde::FdeRuntimeConfigIO::write(serialized, config);
    EXPECT_NE(serialized.str().find("KEDF revapbek\n"), std::string::npos);
}

TEST(FdeRuntimeConfig, RejectsInvalidChargeSpinParity)
{
    std::string text(fluoride_substitution_config());
    const std::string valid = "F 0 1 CH3Cl -1 -1";
    text.replace(text.find(valid), valid.size(), "F 0 0 CH3Cl -1 0");
    std::istringstream input(text);
    EXPECT_THROW(fde::FdeRuntimeConfigIO::read(input), std::invalid_argument);
}

TEST(FdeRuntimeConfig, RejectsUnknownKeywordWithLineNumber)
{
    std::string text(fluoride_substitution_config());
    text.insert(text.find("END_FDE_CONFIG"), "SURPRISE 1\n");
    std::istringstream input(text);
    try
    {
        fde::FdeRuntimeConfigIO::read(input);
        FAIL() << "invalid keyword was accepted";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("FDE_CONFIG line"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("SURPRISE"), std::string::npos);
    }
}
