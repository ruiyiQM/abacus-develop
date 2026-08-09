#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def arguments():
    parser = argparse.ArgumentParser(description="Parse one ABACUS FDE scaling case")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--allocated-cpus-per-rank", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_match(pattern: str, text: str, description: str):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot parse {description}")
    return match


args = arguments()
stdout = (args.case_dir / "abacus.out").read_text(encoding="utf-8")
running = next((args.case_dir / name).read_text(encoding="utf-8")
               for name in ("OUT." + f"fde_scaling_{args.layout}_{args.job_id}" + "/running_scf.log",)
               if (args.case_dir / name).is_file())

iteration = require_match(
    r"^\s*GE1\s+\S+\s+\S+\s+(-?\S+)\s+(-?\S+)\s+(\S+)\s+(\S+)\s*$",
    stdout,
    "first SCF iteration",
)
energy = require_match(r"^\s*E_KohnSham\s+(-?\S+)", running, "Kohn-Sham energy")
total = require_match(r"^\s*TOTAL\s+Time\s+:\s+(\S+)", stdout, "total wall time")

values = {
    "schema": 1,
    "layout": args.layout,
    "job_id": args.job_id,
    "nodes": args.nodes,
    "mpi_ranks": args.ranks,
    "omp_threads_per_rank": args.threads,
    "allocated_cpus_per_rank": args.allocated_cpus_per_rank,
    "total_cores": args.ranks * args.threads,
    "allocated_total_cpus": args.ranks * args.allocated_cpus_per_rank,
    "source_commit": args.source_commit,
    "binary_sha256": args.binary_sha256,
    "etot_ry": float(energy.group(1)),
    "etot_ev_table": float(iteration.group(1)),
    "ediff_ev": float(iteration.group(2)),
    "drho": float(iteration.group(3)),
    "scf_iteration_seconds": float(iteration.group(4)),
    "abacus_total_seconds": float(total.group(1)),
    "input_sha256": sha256(args.case_dir / "INPUT"),
    "stru_sha256": sha256(args.case_dir / "STRU"),
    "kpt_sha256": sha256(args.case_dir / "KPT"),
    "fde_config_sha256": sha256(args.case_dir / "FDE_CONFIG"),
}
if not all(math.isfinite(value) for key, value in values.items()
           if key in {"etot_ry", "etot_ev_table", "ediff_ev", "drho",
                      "scf_iteration_seconds", "abacus_total_seconds"}):
    raise RuntimeError("scaling result contains a non-finite scalar")
print(json.dumps(values, indent=2, sort_keys=True))
