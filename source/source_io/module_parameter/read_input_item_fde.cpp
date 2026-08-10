#include "read_input.h"

#include "read_input_tool.h"
#include "source_base/tool_quit.h"

namespace ModuleIO
{

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
* diabatic_postprocess: before UnitCell setup, assemble determinant overlaps, linearized couplings, and nonorthogonal adiabatic roots from fde_config.)";
        item.default_value = "none";
        item.unit = "";
        item.availability = "LCAO collinear-spin PBE calculations";
        read_sync_string(input.fde_task);
        item.reset_value = [](const Input_Item&, Parameter& para) {
            // A spin-polarized embedded subsystem carries a prescribed
            // alpha/beta population.  In particular, nupdown == 0 means
            // N_alpha == N_beta here, rather than the unconstrained
            // single-Fermi-level behavior of an ordinary UKS calculation.
            // Closed-shell RKS has one doubly occupied spatial-orbital channel
            // and must retain the ordinary single Fermi level.
            if (para.input.fde_task == "embedded_scf")
            {
                para.sys.two_fermi = para.input.nspin == 2;
            }
        };
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            const std::string& task = para.input.fde_task;
            if (task != "none" && task != "embedded_scf" && task != "diabatic_postprocess")
            {
                ModuleBase::WARNING_QUIT(
                    "ReadInput",
                    "fde_task must be none, embedded_scf, or diabatic_postprocess");
            }
            if (task == "none")
            {
                return;
            }
            if (para.input.fde_config.empty())
            {
                ModuleBase::WARNING_QUIT("ReadInput", "fde_config must not be empty when FDE is enabled");
            }
            if (task == "embedded_scf")
            {
                if (para.input.calculation != "scf" || para.input.basis_type != "lcao"
                    || (para.input.nspin != 1 && para.input.nspin != 2)
                    || para.input.noncolin
                    || para.input.lspinorb)
                {
                    ModuleBase::WARNING_QUIT(
                        "ReadInput",
                        "fde_task embedded_scf requires calculation scf, basis_type lcao, "
                        "nspin 1 or 2, noncolin 0, and lspinorb 0");
                }
                if (para.input.dft_functional != "pbe")
                {
                    ModuleBase::WARNING_QUIT(
                        "ReadInput",
                        "fde_task embedded_scf currently requires dft_functional pbe");
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
        item.description = R"(Path to the deterministic FDE_CONFIG sidecar file. The sidecar defines fragments, diabatic charge/spin states, density/determinant/linearized-state artifacts, convergence controls, and coupling/diagonalization selections. Relative paths are resolved from the ABACUS working directory.)";
        item.default_value = "FDE_CONFIG";
        item.unit = "";
        item.availability = "fde_task is not none";
        read_sync_string(input.fde_config);
        this->add_item(item);
    }
}

} // namespace ModuleIO
