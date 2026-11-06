"""Exercise native CLI rejection and complex postprocessing under MPI."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile


def write_complex_fixture(root):
    # Same independent two-state one-body model as test_fde_diabatic_postprocess.cpp.
    identity = ("GEOMETRY fixed_cell\nORBITALS fixed_basis\n"
                "BOUNDARY_TWIST 0.17 0.11 0\nRESOURCES fixed_resources\n")
    (root / "overlap").write_text(
        "FDE_AO_MATRIX 2\n" + identity + "DIMENSION 3\nVALUES 9 "
        "(1,0) (0,-0.8) (0,0) (0,0.8) (1,0) (0,0) (0,0) (0,0) (1,0)\nEND\n")
    for label, energy, coefficients in (("D", 5.0, "(1,0) (0,0) (0,0)"),
                                         ("A", 4.4, "(0,0) (1,0) (0,0)")):
        (root / f"{label}.det").write_text(
            f"FDE_DIABATIC_DETERMINANT 2\nSTATE {label}\n" + identity +
            f"AO_DIMENSION 3\nALPHA 1\nORBITAL {label} -1 {coefficients}\n"
            "BETA 1\nORBITAL A -2 (0,0) (0,0) (1,0)\nEND\n")
        (root / f"{label}.linear").write_text(
            f"FDE_LINEARIZED_STATE 2\nSTATE {label}\n" + identity +
            f"AO_DIMENSION 3\nREFERENCE_ENERGY_RY {energy}\n"
            "HAMILTONIAN_ALPHA_RY 9 (2,0) (-0.3,-1.45) (0,0) "
            "(-0.3,1.45) (1.4,0) (0,0) (0,0) (0,0) (0,0)\n"
            "HAMILTONIAN_BETA_RY 9 (3,0) (0,0) (0,0) "
            "(0,0) (3,0) (0,0) (0,0) (0,0) (3,0)\nEND\n")
    (root / "FDE_CONFIG").write_text(
        "FDE_CONFIG 1\nATOM_COUNT 2\nFRAGMENT D 1 1 0\nFRAGMENT A 2 1 1\n"
        "STATE D 1 0 2 D 0 1 A 1 -1\nSTATE A 1 0 2 D 1 0 A 0 0\n"
        "DETERMINANT D D.det\nDETERMINANT A A.det\n"
        "LINEARIZED_STATE D D.linear\nLINEARIZED_STATE A A.linear\n"
        "DIAGONAL_ENERGY_RY D 5\nDIAGONAL_ENERGY_RY A 4.4\nAO_OVERLAP overlap\n"
        "BOUNDARY_TWIST 0.17 0.11 0\nOUTPUT_PREFIX result\nEND_FDE_CONFIG\n")
    (root / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nfde_task diabatic_postprocess\n"
        "fde_config FDE_CONFIG\n")


def check_complex_postprocess(executable, launcher, directory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools/fde"))
    from fde_workflow import read_postprocess_result

    reports = []
    for failure in (None, "result", "table"):
        root = directory / (failure or "success")
        root.mkdir()
        write_complex_fixture(root)
        if failure:
            suffix = ".fde_diabatic" if failure == "result" else ".fde_diabatic.tsv"
            (root / ("result" + suffix)).mkdir()
        command = launcher + [str(executable)]
        completed = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=20)
        report = {"case": failure or "success", "command": command,
                  "exit_code": completed.returncode,
                  "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in ("INPUT", "FDE_CONFIG", "overlap", "D.det", "A.det",
                                                "D.linear", "A.linear")}}
        if failure:
            expected = f"Cannot create FDE diabatic {failure}:"
            assert completed.returncode == 1 and expected in completed.stdout, completed.stdout
            assert "FDE diabatic postprocess completed" not in completed.stdout, completed.stdout
        else:
            assert completed.returncode == 0, completed.stdout
            assert completed.stdout.count("FDE diabatic postprocess completed") == 1, completed.stdout
            path = root / "result.fde_diabatic"
            result = read_postprocess_result(path, ["D", "A"])
            assert result["artifact_version"] == 2 and result["boundary_twist"] == [0.17, 0.11, 0.0]
            pair = result["pairs"][0]
            for name, expected in (("overlap", [0.0, 0.8]),
                                   ("h12_ry", [-0.3, 3.85]),
                                   ("orthogonalized_coupling_ry", [-0.5, 0.25])):
                assert all(abs(a - b) < 1e-12 for a, b in zip(pair[name], expected)), pair
            assert all(abs(a - b) < 1e-12 for a, b in
                       zip(result["adiabatic_energies_ry"], [3.75, 5.25])), result
            confirmations = {}
            for suffix in (".fde_diabatic", ".fde_diabatic.tsv"):
                notice = "Wrote file: " + str((root / ("result" + suffix)).resolve())
                count = sum(line == notice for log in (root / "OUT.ABACUS").rglob("*.log")
                            for line in log.read_text().splitlines())
                assert count == 1, (notice, count)
                confirmations[suffix] = count
            report.update(result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          writer_confirmations=confirmations, result=result)
        reports.append(report)
    print(json.dumps({"passed": True, "postprocess_v2": reports,
                      "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}))


def main():
    executable = Path(sys.argv[1]).resolve()
    launcher = sys.argv[2:]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "FDE_CONFIG").write_text(
            "FDE_CONFIG 1\nATOM_COUNT 2\n"
            "FRAGMENT A 1 1 0\nFRAGMENT B 1 1 1\n"
            "STATE first 0 0 2 A 0 1 B 0 -1\n"
            "STATE second 0 0 2 A 0 -1 B 0 1\n"
            "CALCULATE_FORCE true\nEND_FDE_CONFIG\n", encoding="utf-8")
        for force, stress in ((0, 0), (1, 0), (0, 1)):
            (root / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\n"
                "fde_task diabatic_postprocess\nfde_config FDE_CONFIG\n"
                f"cal_force {force}\ncal_stress {stress}\n", encoding="utf-8")
            result = subprocess.run(
                launcher + [str(executable)], cwd=root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            expected = ("native FDE forces and stress are not supported"
                        if force or stress else "CALCULATE_FORCE must be false")
            if result.returncode != 1 or expected not in result.stdout:
                raise AssertionError(f"exit={result.returncode}; expected {expected}\n{result.stdout}")
        check_complex_postprocess(executable, launcher, root)


if __name__ == "__main__":
    main()
