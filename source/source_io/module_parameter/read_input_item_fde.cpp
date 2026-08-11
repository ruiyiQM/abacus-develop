#include "read_input.h"

#include "fde_functional_capability.h"
#include "read_input_tool.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <cctype>

namespace ModuleIO
{

FdeXcCapability classify_fde_xc_functional(const std::string& functional)
{
    std::string normalized(functional);
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](const unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    if (normalized == "pbe")
    {
        return FdeXcCapability::pbe_semilocal;
    }
    if (normalized == "scan0" || normalized.find("hyb_mgga_") != std::string::npos)
    {
        return FdeXcCapability::hybrid_meta_gga_requires_tau_and_exact_exchange;
    }
    if (normalized == "hf" || normalized == "pbe0" || normalized == "hse"
        || normalized == "b3lyp" || normalized == "lc_pbe"
        || normalized == "lc_wpbe" || normalized == "lrc_wpbe"
        || normalized == "lrc_wpbeh" || normalized == "cam_pbeh"
        || normalized == "wp22" || normalized == "cwp22"
        || normalized == "muller" || normalized == "power"
        || normalized.find("hyb_lda_") != std::string::npos
        || normalized.find("hyb_gga_") != std::string::npos)
    {
        return FdeXcCapability::hybrid_requires_exact_exchange;
    }
    if (normalized == "scan" || normalized.find("mgga_") != std::string::npos)
    {
        return FdeXcCapability::meta_gga_requires_tau;
    }
    return FdeXcCapability::unsupported;
}

bool supports_fde_fragment_xc(const std::string& functional)
{
    std::string normalized(functional);
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](const unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return normalized == "pbe" || normalized == "pbe0" || normalized == "scan";
}

std::string fde_xc_capability_error(const FdeXcCapability capability)
{
    if (capability
        == FdeXcCapability::hybrid_meta_gga_requires_tau_and_exact_exchange)
    {
        return "hybrid meta-GGA is a valid fragment solver class, but the "
               "native FDE workflow currently exposes only pbe, pbe0, and scan "
               "as fragment_xc choices";
    }
    if (capability == FdeXcCapability::hybrid_requires_exact_exchange)
    {
        return "hybrid XC is evaluated only inside each fragment; use pbe0 as "
               "fragment_xc and pbe as embedding_xc";
    }
    if (capability == FdeXcCapability::meta_gga_requires_tau)
    {
        return "meta-GGA XC is evaluated only inside each fragment; use scan "
               "as fragment_xc and pbe as embedding_xc";
    }
    return "fde_task embedded_scf fragment XC must be pbe, pbe0, or scan";
}

void ReadInput::item_fde()
{
    {
        Input_Item item("fde_task");
        item.annotation = "native frozen-density embedding runtime task";
        item.category = "Frozen-density embedding";
        item.type = "String";
        item.description = R"(Select the native FDE runtime entry point.
* none: run an ordinary ABACUS calculation.
* embedded_scf: run one subsystem-in-environment LCAO SCF job described by fde_config.
* embedded_session: keep one fixed subsystem worker alive and accept RUN requests on standard input.
* diabatic_postprocess: before UnitCell setup, assemble determinant overlaps, linearized couplings, and nonorthogonal adiabatic roots from fde_config.)";
        item.default_value = "none";
        item.unit = "";
        item.availability = "LCAO collinear-spin PBE, PBE0, or SCAN fragment calculations";
        read_sync_string(input.fde_task);
        item.reset_value = [](const Input_Item&, Parameter& para) {
            // A spin-polarized embedded subsystem carries a prescribed
            // alpha/beta population.  In particular, nupdown == 0 means
            // N_alpha == N_beta here, rather than the unconstrained
            // single-Fermi-level behavior of an ordinary UKS calculation.
            // Closed-shell RKS has one doubly occupied spatial-orbital channel
            // and must retain the ordinary single Fermi level.
            if (para.input.fde_task == "embedded_scf"
                || para.input.fde_task == "embedded_session")
            {
                para.sys.two_fermi = para.input.nspin == 2;
            }
        };
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            const std::string& task = para.input.fde_task;
            if (task != "none" && task != "embedded_scf"
                && task != "embedded_session"
                && task != "diabatic_postprocess")
            {
                ModuleBase::WARNING_QUIT(
                    "ReadInput",
                    "fde_task must be none, embedded_scf, embedded_session, "
                    "or diabatic_postprocess");
            }
            if (task == "none")
            {
                return;
            }
            if (para.input.fde_config.empty())
            {
                ModuleBase::WARNING_QUIT("ReadInput", "fde_config must not be empty when FDE is enabled");
            }
            if (task == "embedded_scf" || task == "embedded_session")
            {
                if (para.input.calculation != "scf" || para.input.basis_type != "lcao"
                    || (para.input.nspin != 1 && para.input.nspin != 2)
                    || para.input.noncolin
                    || para.input.lspinorb)
                {
                    ModuleBase::WARNING_QUIT(
                        "ReadInput",
                        "fde_task embedded_scf/embedded_session requires calculation scf, basis_type lcao, "
                        "nspin 1 or 2, noncolin 0, and lspinorb 0");
                }
                if (!supports_fde_fragment_xc(para.input.dft_functional))
                {
                    ModuleBase::WARNING_QUIT(
                        "ReadInput",
                        fde_xc_capability_error(
                            classify_fde_xc_functional(
                                para.input.dft_functional)));
                }
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("fde_config");
        item.annotation = "path to the native FDE sidecar configuration";
        item.category = "Frozen-density embedding";
        item.type = "String";
        item.description = R"(Path to the deterministic FDE_CONFIG sidecar file. The sidecar defines fragments, diabatic charge/spin states, independent fragment/embedding XC choices, density/determinant/linearized-state artifacts, convergence controls, and coupling/diagonalization selections. Relative paths are resolved from the ABACUS working directory.)";
        item.default_value = "FDE_CONFIG";
        item.unit = "";
        item.availability = "fde_task is not none";
        read_sync_string(input.fde_config);
        this->add_item(item);
    }
}

} // namespace ModuleIO
