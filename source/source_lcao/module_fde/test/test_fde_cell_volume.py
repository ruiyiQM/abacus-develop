"""Check native cell-volume validation and authoritative output metadata."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2]).resolve()
    resources = repository / "tests/PP_ORB"
    sys.path.insert(0, str(repository / "tools/fde"))
    from fde_workflow import read_density

    with tempfile.TemporaryDirectory(prefix="fde-cell-volume-") as directory:
        for name, volume, rejected in (("matching", 4096.0, False),
                                       ("rounded", 4096.0 * (1 + 3e-6), True),
                                       ("nearby", 4096.0 * (1 + 1e-12), False)):
            root = Path(directory) / name
            root.mkdir()
            (root / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\n"
                "gamma_only 1\necutwfc 10\nnx 40\nny 40\nnz 40\n"
                "nspin 2\nnelec 1\nnupdown 1\nnbands 1\nsymmetry 0\n"
                "dft_functional pbe\nks_solver lapack\nkpar 1\n"
                "smearing_method fixed\nscf_nmax 1\nscf_thr 1e-8\n"
                "fde_task embedded_scf\nfde_config FDE_CONFIG\n"
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
                    "GEOMETRY pair\nGRID_FINGERPRINT grid-40x40x40\n"
                    "PSEUDOPOTENTIALS hydrogen\nORBITALS hydrogen-1s\n"
                    "CORE_DENSITY none\nFUNCTIONALS pbe pw91k\n"
                    f"GRID 40 40 40 {volume:.17g}\nPOPULATIONS {alpha} {beta}\n"
                    f"RHO_UNIFORM {alpha / volume:.17g} {beta / volume:.17g}\nEND\n",
                    encoding="utf-8")
            (root / "FDE_CONFIG").write_text(
                "FDE_CONFIG 1\nATOM_COUNT 2\nFRAGMENT A 1 1 0\nFRAGMENT B 1 1 1\n"
                "STATE first 0 0 2 A 0 1 B 0 -1\n"
                "STATE second 0 0 2 A 0 -1 B 0 1\nACTIVE_STATE first\n"
                "ACTIVE_FRAGMENT A\nACTIVE_DENSITY A.fde_seed\n"
                "FROZEN_DENSITY B B.fde_seed\nOUTPUT_PREFIX result\n"
                "MAX_SCF_ITERATIONS 1\nSCF_DENSITY_TOLERANCE 1e-8\nEND_FDE_CONFIG\n",
                encoding="utf-8")
            result = subprocess.run([str(executable)], cwd=root, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            diagnostic = "density artifact cell volume does not match the ABACUS unit cell"
            if rejected:
                assert result.returncode == 1 and diagnostic in result.stdout, result.stdout
            else:
                assert result.returncode == 0, result.stdout
                path = root / "result.fde_density"
                if not path.exists():
                    path = root / "result.partial.fde_density"
                assert read_density(path)["cell_volume"] == 4096.0
                config = root / "FDE_CONFIG"
                config.write_text(config.read_text().replace("ACTIVE_DENSITY A.fde_seed",
                                                          f"ACTIVE_DENSITY {path}")
                                  .replace("OUTPUT_PREFIX result", "OUTPUT_PREFIX repeat"), encoding="utf-8")
                repeated = subprocess.run([str(executable)], cwd=root, text=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
                assert repeated.returncode == 0, repeated.stdout

        root = Path(directory) / "matching"
        input_path = root / "INPUT"
        input_path.write_text(input_path.read_text().replace("embedded_scf", "embedded_session"), encoding="utf-8")
        config = root / "FDE_CONFIG"
        bad = root / "FDE_CONFIG_BAD"
        bad.write_text(config.read_text().replace("FROZEN_DENSITY B B.fde_seed",
                                                 "FROZEN_DENSITY B ../rounded/B.fde_seed")
                       .replace("ACTIVE_DENSITY " + str(root / "result.partial.fde_density"),
                                "ACTIVE_DENSITY ../rounded/A.fde_seed"), encoding="utf-8")
        result = subprocess.run([str(executable)], cwd=root,
                                input=f"RUN valid {config}\nRUN invalid {bad}\nSTOP\n", text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        assert "FDE_SESSION_DONE valid " in result.stdout, result.stdout
        assert "FDE_SESSION_ERROR invalid " in result.stdout and diagnostic in result.stdout, result.stdout
    print("Native FDE rejects mismatched volumes and exports the actual unit-cell volume.")


if __name__ == "__main__":
    main()
