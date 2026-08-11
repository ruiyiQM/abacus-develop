#!/usr/bin/env python3
"""Replace local launch commands in an FDE workflow with Euler Slurm steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def launcher(
    srun: str,
    ranks: int,
    threads: int,
    binary: Path,
    *,
    exclusive: bool,
    overlap: bool,
) -> List[str]:
    command = [srun]
    if exclusive:
        command.append("--exclusive")
    if overlap:
        command.extend(("--overlap", "--exact"))
    command.extend(
        [
            "--cpu-bind=cores",
            "--distribution=block:block",
            f"--ntasks={ranks}",
            f"--cpus-per-task={threads}",
            str(binary),
        ]
    )
    return command


def configure(
    spec: Dict[str, Any],
    *,
    binary: Path,
    ranks: int,
    threads: int,
    postprocess_ranks: int,
    srun: str,
    exclusive: bool,
    persistent_session: bool,
    work_directory: Path | None,
    device: str | None,
    ks_solver: str | None,
    binary_sha256: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("workflow root must be a JSON object")
    if "controls" not in spec or not isinstance(spec["controls"], dict):
        raise ValueError("workflow controls must be a JSON object")
    configured = dict(spec)
    configured["controls"] = dict(spec["controls"])
    configured["abacus_command"] = launcher(
        srun, ranks, threads, binary, exclusive=exclusive, overlap=False
    )
    configured["postprocess_command"] = launcher(
        srun, postprocess_ranks, threads, binary,
        exclusive=exclusive, overlap=False
    )
    if persistent_session:
        configured["controls"]["execution_mode"] = "persistent_session"
        configured["session_command"] = launcher(
            srun, ranks, threads, binary, exclusive=False, overlap=True
        )
    if work_directory is not None:
        configured["work_directory"] = str(work_directory)
    if device is not None:
        configured["controls"]["device"] = device
    if ks_solver is not None:
        configured["controls"]["ks_solver"] = ks_solver

    provenance = configured.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("workflow provenance must be a JSON object when present")
    provenance = dict(provenance)
    provenance["euler_slurm"] = {
        "ranks_per_subsystem": ranks,
        "threads_per_rank": threads,
        "postprocess_ranks": postprocess_ranks,
        "exclusive_steps": exclusive,
        "persistent_session": persistent_session,
        "binary": str(binary),
    }
    if binary_sha256 is not None:
        provenance["euler_slurm"]["binary_sha256"] = binary_sha256
    configured["provenance"] = provenance
    return configured


def write_json_atomic(path: Path, payload: Dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"output exists; pass --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path, help="input workflow JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--ranks", type=positive_integer, default=20)
    parser.add_argument("--threads", type=positive_integer, default=4)
    parser.add_argument("--postprocess-ranks", type=positive_integer, default=1)
    parser.add_argument("--srun", default="srun")
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="make each Slurm step exclusive (required for concurrent Jacobi steps)",
    )
    parser.add_argument(
        "--persistent-session",
        action="store_true",
        help="reuse fixed-fragment workers with overlapping resident Slurm steps",
    )
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--ks-solver")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    workflow = args.workflow.resolve(strict=True)
    binary = args.abacus.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"ABACUS path is not an executable file: {binary}")
    if any(character.isspace() for character in str(binary)):
        raise ValueError("ABACUS executable path must not contain whitespace")
    if args.device == "gpu" and args.ks_solver is None:
        args.ks_solver = "cusolver"
    if args.device == "gpu" and args.ks_solver not in ("cusolver", "elpa"):
        raise ValueError("Euler GPU workflows require cusolver or GPU-enabled elpa")

    spec = json.loads(workflow.read_text(encoding="utf-8"))
    work_directory = (
        args.work_directory.expanduser().resolve()
        if args.work_directory is not None
        else None
    )
    configured = configure(
        spec,
        binary=binary,
        ranks=args.ranks,
        threads=args.threads,
        postprocess_ranks=args.postprocess_ranks,
        srun=args.srun,
        exclusive=args.exclusive,
        persistent_session=args.persistent_session,
        work_directory=work_directory,
        device=args.device,
        ks_solver=args.ks_solver,
        binary_sha256=sha256(binary),
    )
    write_json_atomic(args.output.expanduser().resolve(), configured, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
