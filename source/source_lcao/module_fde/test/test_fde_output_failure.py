"""Mandatory FDE output failures must end a request on every MPI rank."""

from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile


def main():
    executable = Path(sys.argv[1]).resolve()
    resources = Path(sys.argv[2]).resolve() / "tests/PP_ORB"
    launcher = sys.argv[3:]
    solver = "scalapack_gvx" if launcher else "lapack"
    with tempfile.TemporaryDirectory(prefix="fde-output-failure-") as directory:
        for gamma, failure in ((1, "density"), (0, "density"),
                               (1, "band"), (0, "band"), (1, "fragment"),
                               (0, "fragment"), (0, "complex-control"),
                               (1, "restart-open"), (1, "restart-flush")):
            if failure == "restart-flush" and not Path("/dev/full").exists():
                continue
            root = Path(directory) / f"gamma{gamma}-{failure}"
            root.mkdir()
            restart = failure.startswith("restart-")
            finite_twist = gamma == 0 and failure in ("fragment", "complex-control")
            cell = 8 if finite_twist else 16
            volume = cell ** 3
            # Band/fragment cases need to reach their converged-output writers.
            tolerance = 10.0 if failure in ("band", "fragment", "complex-control") else 1e-8
            out_chg = 0 if restart else -1
            (root / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\n"
                f"gamma_only {gamma}\necutwfc 10\nnx 40\nny 40\nnz 40\n"
                "nspin 2\nnelec 1\nnupdown 1\nnbands 2\nsymmetry 0\n"
                f"dft_functional pbe\nks_solver {solver}\nkpar 1\n"
                f"smearing_method fixed\nscf_nmax 1\nscf_thr {tolerance}\n"
                f"out_chg {out_chg}\nfde_task embedded_session\nfde_config FDE_CONFIG\n"
                f"pseudo_dir {resources}\norbital_dir {resources}\n", encoding="utf-8")
            kpoints = ("K_POINTS\n1\nDirect\n0.17 0.11 0 1\n" if finite_twist
                       else "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n")
            if gamma == 0 and failure == "band":
                # General periodic SCF retains its multi-k density/band path without a finite determinant.
                kpoints = "K_POINTS\n0\nGamma\n3 1 1 0 0 0\n"
            (root / "KPT").write_text(kpoints, encoding="utf-8")
            (root / "STRU").write_text(
                "ATOMIC_SPECIES\nH 1 H_ONCV_PBE-1.0.upf\n"
                "NUMERICAL_ORBITAL\nH_gga_8au_100Ry_2s1p.orb\n"
                f"LATTICE_CONSTANT\n1\nLATTICE_VECTORS\n{cell} 0 0\n0 {cell} 0\n0 0 {cell}\n"
                "ATOMIC_POSITIONS\nDirect\nH\n0\n2\n"
                "0.25 0.5 0.5 0 0 0\n0.75 0.5 0.5 0 0 0\n", encoding="utf-8")
            for label, alpha, beta in (("A", 1, 0), ("B", 0, 1)):
                (root / f"{label}.fde_seed").write_text(
                    f"FDE_UNIFORM_DENSITY_SEED 1\nFRAGMENT {label}\nSTATE first\n"
                    "GEOMETRY pair\nGRID_FINGERPRINT grid-40x40x40\n"
                    "PSEUDOPOTENTIALS hydrogen\nORBITALS hydrogen-2s1p\n"
                    "CORE_DENSITY none\nFUNCTIONALS pbe pw91k\n"
                    f"GRID 40 40 40 {volume}\nPOPULATIONS {alpha} {beta}\n"
                    f"RHO_UNIFORM {alpha / volume:.17g} {beta / volume:.17g}\nEND\n",
                    encoding="utf-8")
            prefix = "result"
            if restart:
                output = root / "OUT.ABACUS"
                output.mkdir()
                path = output / "ABACUS-CHARGE-DENSITY.restart"
                if failure == "restart-open":
                    path.mkdir()
                else:
                    path.symlink_to("/dev/full")
            elif failure == "density":
                (root / "blocked").write_text("not a directory\n", encoding="utf-8")
                prefix = "blocked/result"
                diagnostic = "Cannot create FDE density artifact:"
            elif failure == "complex-control":
                pass
            else:
                suffix = ".fde_kbands" if failure == "band" else ".fde_fragment"
                (root / (prefix + suffix)).mkdir()
                diagnostic = ("Cannot create FDE k-point band artifact:" if failure == "band"
                              else "Cannot create FDE fragment SCF artifact:")
            config = root / "FDE_CONFIG"
            boundary = "BOUNDARY_TWIST 0.17 0.11 0\n" if finite_twist else ""
            config.write_text(
                "FDE_CONFIG 1\nATOM_COUNT 2\nFRAGMENT A 1 1 0\nFRAGMENT B 1 1 1\n"
                "STATE first 0 0 2 A 0 1 B 0 -1\n"
                "STATE second 0 0 2 A 0 -1 B 0 1\nACTIVE_STATE first\n"
                "ACTIVE_FRAGMENT A\nACTIVE_DENSITY A.fde_seed\n"
                f"FROZEN_DENSITY B B.fde_seed\nOUTPUT_PREFIX {prefix}\n"
                + boundary +
                f"MAX_SCF_ITERATIONS 1\nSCF_DENSITY_TOLERANCE {tolerance}\nEND_FDE_CONFIG\n",
                encoding="utf-8")
            result = subprocess.run(
                launcher + [str(executable)], cwd=root, input=f"RUN failure {config}\nSTOP\n",
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            if failure == "complex-control":
                assert result.returncode == 0, result.stdout
                assert "FDE_SESSION_DONE failure " in result.stdout, result.stdout
                assert "FDE_SESSION_ERROR" not in result.stdout, result.stdout
                rows = [line.split() for line in (root / "result.fde_fragment").read_text().splitlines()]
                assert rows[0] == ["FDE_FRAGMENT_SCF_ARTIFACT", "2"], rows[0]
                records = {fields[0]: fields[1:] for fields in rows if fields}
                assert [float(value) for value in records["BOUNDARY_TWIST"]] == [0.17, 0.11, 0.0]
                assert records["RESOURCES"] and records["ALPHA"] == ["1"] and records["BETA"] == ["0"]
                maximum_imaginary = 0.0
                for name in ("AO_OVERLAP", "HAMILTONIAN_ALPHA_RY", "HAMILTONIAN_BETA_RY"):
                    assert int(records[name][0]) == len(records[name]) - 1
                    for token in records[name][1:]:
                        real, imaginary = token.strip("()").split(",")
                        assert math.isfinite(float(real)) and math.isfinite(float(imaginary))
                        maximum_imaginary = max(maximum_imaginary, abs(float(imaginary)))
                assert maximum_imaginary > 1e-12, "non-Gamma output unexpectedly reduced to real matrices"
                written_notice = "Wrote file: " + str((root / "result.fde_fragment").resolve())
                confirmations = sum(line == written_notice
                                    for path in (root / "OUT.ABACUS").rglob("*.log")
                                    for line in path.read_text().splitlines())
                assert confirmations == 1, ("expected one root writer", confirmations)
                print(json.dumps({"complex_artifact_control": "passed", "twist": [0.17, 0.11, 0.0],
                                  "maximum_matrix_imaginary": maximum_imaginary,
                                  "writer_confirmations": confirmations,
                                  "scope": "one-step IO control; not freeze-thaw or scientific acceptance"}))
            elif restart:
                assert result.returncode == 0, result.stdout
                assert "FDE_SESSION_DONE failure " in result.stdout, result.stdout
                assert "FDE_SESSION_ERROR" not in result.stdout, result.stdout
                warning = (output / "warning.log").read_text(encoding="utf-8")
                assert "File I/O failure" in warning and str(path.relative_to(root)) in warning, warning
            else:
                assert result.returncode == 1, result.stdout
                assert "FDE_SESSION_ERROR failure " in result.stdout, result.stdout
                assert diagnostic in result.stdout, result.stdout
                assert "FDE_SESSION_DONE failure " not in result.stdout, result.stdout
    print("Native FDE output failures return a request error without an MPI hang.")


if __name__ == "__main__":
    main()
