"""Check independent JSON records for two native persistent FDE requests."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = Path(sys.argv[1]).resolve()
    resources = Path(sys.argv[2]).resolve() / "tests/PP_ORB"
    with tempfile.TemporaryDirectory(prefix="fde-session-json-") as directory:
        root = Path(directory)
        (root / "INPUT").write_text(
            "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\n"
            "gamma_only 1\necutwfc 10\nnx 40\nny 40\nnz 40\n"
            "nspin 2\nnelec 1\nnupdown 1\nnbands 1\nsymmetry 0\n"
            "dft_functional pbe\nks_solver lapack\nkpar 1\n"
            "smearing_method fixed\nmixing_type plain\nmixing_beta 0.1\n"
            "scf_nmax 1\nscf_thr 1e-8\n"
            "fde_task embedded_session\nfde_config FDE_CONFIG\n"
            f"pseudo_dir {resources}\norbital_dir {resources}\n", encoding="utf-8")
        (root / "KPT").write_text("K_POINTS\n0\nGamma\n1 1 1 0 0 0\n", encoding="utf-8")
        (root / "STRU").write_text(
            "ATOMIC_SPECIES\nH 1 H_ONCV_PBE-1.0.upf\n"
            "NUMERICAL_ORBITAL\nH_gga_8au_100Ry_1s.orb\n"
            "LATTICE_CONSTANT\n1\nLATTICE_VECTORS\n16 0 0\n0 16 0\n0 0 16\n"
            "ATOMIC_POSITIONS\nDirect\nH\n0\n2\n"
            "0.25 0.5 0.5 0 0 0\n0.75 0.5 0.5 0 0 0\n", encoding="utf-8")
        for label, alpha, beta in (("A", 1, 0), ("B", 0, 1)):
            (root / f"{label}.fde_seed").write_text(
                f"FDE_UNIFORM_DENSITY_SEED 1\nFRAGMENT {label}\nSTATE first\n"
                "GEOMETRY hydrogen-pair\nGRID_FINGERPRINT grid-40x40x40\n"
                "PSEUDOPOTENTIALS hydrogen\nORBITALS hydrogen-1s\n"
                "CORE_DENSITY none\nFUNCTIONALS pbe pw91k\n"
                f"GRID 40 40 40 4096\nPOPULATIONS {alpha} {beta}\n"
                f"RHO_UNIFORM {alpha / 4096:.17g} {beta / 4096:.17g}\nEND\n",
                encoding="utf-8")
        config = root / "FDE_CONFIG"
        config.write_text(
            "FDE_CONFIG 1\nATOM_COUNT 2\nFRAGMENT A 1 1 0\nFRAGMENT B 1 1 1\n"
            "STATE first 0 0 2 A 0 1 B 0 -1\nSTATE second 0 0 2 A 0 -1 B 0 1\n"
            "ACTIVE_STATE first\nACTIVE_FRAGMENT A\nACTIVE_DENSITY A.fde_seed\n"
            "FROZEN_DENSITY B B.fde_seed\nOUTPUT_PREFIX result\n"
            "MAX_SCF_ITERATIONS 1\nSCF_DENSITY_TOLERANCE 1e-8\nEND_FDE_CONFIG\n",
            encoding="utf-8")
        result = subprocess.run(
            [str(executable)], cwd=root,
            input=f"RUN first {config}\nRUN second {config}\nSTOP\n",
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
        if (result.returncode != 0 or "FDE_SESSION_ERROR" in result.stdout
                or "FDE_SESSION_DONE first " not in result.stdout
                or "FDE_SESSION_DONE second " not in result.stdout):
            raise AssertionError(f"exit={result.returncode}\n{result.stdout}")
        outputs = json.loads((root / "abacus.json").read_text(encoding="utf-8"))["output"]
        assert len(outputs) == 2, outputs
        for record in outputs:
            assert len(record["scf"]) == 1, record
            assert math.isfinite(record["scf"][0]["energy"]), record
            assert math.isfinite(record["scf"][0]["drho"]), record
        print("Two native FDE requests produced separate JSON SCF records.")


if __name__ == "__main__":
    main()
