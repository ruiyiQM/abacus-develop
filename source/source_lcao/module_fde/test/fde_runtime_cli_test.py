"""Reject unsupported native force requests with an ordinary CLI error."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = Path(sys.argv[1]).resolve()
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
                [str(executable)], cwd=root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            expected = ("native FDE forces and stress are not supported"
                        if force or stress else "CALCULATE_FORCE must be false")
            if result.returncode != 1 or expected not in result.stdout:
                raise AssertionError(f"exit={result.returncode}; expected {expected}\n{result.stdout}")


if __name__ == "__main__":
    main()
