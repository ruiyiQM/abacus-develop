"""Check fatal MPI exits independently of launcher termination policy."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    executable = Path(sys.argv[1]).resolve()
    ranks = int(sys.argv[2])
    launcher = sys.argv[3:]
    assert ranks in (1, 2, 4) and launcher
    modes = ["guards", "warning_root", "check_root", "quit_root"]
    if ranks > 1:
        modes.append("warning_last")
    records = []
    with tempfile.TemporaryDirectory(prefix="abacus-fatal-mpi-") as directory:
        for mode in modes:
            command = launcher + [str(executable), mode]
            started = time.monotonic()
            result = subprocess.run(command, cwd=directory, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
            expected = 0 if mode == "guards" else 1
            assert result.returncode == expected, (mode, result.returncode, result.stdout)
            if mode == "guards":
                assert result.stdout.count("ABACUS_MPI_GUARDS_OK") == ranks, result.stdout
            else:
                failing_rank = ranks - 1 if mode == "warning_last" else 0
                marker = f"ABACUS_FATAL_PROBE rank={failing_rank} mode={mode}"
                assert marker in result.stdout, result.stdout
            records.append({"mode": mode, "exit_code": result.returncode,
                            "elapsed_seconds": time.monotonic() - started,
                            "deadline_seconds": 15, "command": command})
    print(json.dumps({"passed": True, "ranks": ranks, "records": records,
                      "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
