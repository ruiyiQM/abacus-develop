#!/usr/bin/env python3
"""Summarize and compare FDE workflow performance JSON reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence


class PerformanceSummaryError(RuntimeError):
    """Raised when a workflow performance report is malformed."""


PHASE_NAMES = (
    "workflow_preparation",
    "session_startup",
    "electronic_steps",
    "abacus_overhead",
    "artifact_validation",
)


def nonnegative_number(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name, 0.0)
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) < 0.0):
        raise PerformanceSummaryError(
            f"{name} must be a finite nonnegative number")
    return float(value)


def summarize_report(path: Path, label: str) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") not in (1, 2):
        raise PerformanceSummaryError(
            f"unsupported FDE performance schema in {path}")
    wall = nonnegative_number(payload, "total_wall_time_seconds")
    electronic = nonnegative_number(
        payload, "total_electronic_step_time_seconds")
    calls = int(nonnegative_number(payload, "total_subsystem_calls"))
    iterations = int(nonnegative_number(payload, "total_scf_iterations"))
    phases_payload = payload.get("phase_totals_seconds", {})
    if not isinstance(phases_payload, dict):
        raise PerformanceSummaryError(
            f"phase_totals_seconds must be an object in {path}")
    phases = {
        name: nonnegative_number(phases_payload, name)
        for name in PHASE_NAMES
    }
    profiled = math.fsum(phases.values())
    retries = int(nonnegative_number(payload, "total_retries"))
    reuse = nonnegative_number(payload, "session_reuse_fraction")
    if reuse > 1.0:
        raise PerformanceSummaryError(
            f"session_reuse_fraction exceeds one in {path}")
    return {
        "label": label,
        "path": str(path.resolve()),
        "schema_version": int(payload["schema_version"]),
        "wall_time_seconds": wall,
        "electronic_step_time_seconds": electronic,
        "subsystem_calls": calls,
        "scf_iterations": iterations,
        "total_retries": retries,
        "session_reuse_fraction": reuse,
        "phase_totals_seconds": phases,
        "profiled_time_seconds": profiled,
        "profiled_fraction_of_wall": profiled / wall if wall > 0.0 else 0.0,
    }


def compare_reports(paths: Sequence[Path], labels: Sequence[str]) -> Dict[str, object]:
    if not paths:
        raise PerformanceSummaryError("at least one performance report is required")
    if len(paths) != len(labels):
        raise PerformanceSummaryError("path and label counts differ")
    reports = [summarize_report(path, label)
               for path, label in zip(paths, labels)]
    baseline_wall = float(reports[0]["wall_time_seconds"])
    for index, report in enumerate(reports):
        wall = float(report["wall_time_seconds"])
        report["speedup_vs_baseline"] = (
            baseline_wall / wall if wall > 0.0 else None)
        report["wall_time_delta_seconds"] = wall - baseline_wall
        report["is_baseline"] = index == 0
    return {
        "schema_version": 1,
        "baseline": reports[0]["label"],
        "reports": reports,
    }


def render_tsv(summary: Mapping[str, object]) -> str:
    lines = [
        "label\twall_s\tspeedup\tcalls\tscf_iterations\tretries\t"
        "session_reuse\telectronic_s\tabacus_overhead_s\tsession_startup_s"
    ]
    for report in summary["reports"]:
        speedup = report["speedup_vs_baseline"]
        phases = report["phase_totals_seconds"]
        lines.append("\t".join((
            str(report["label"]),
            f"{float(report['wall_time_seconds']):.6g}",
            "" if speedup is None else f"{float(speedup):.6g}",
            str(report["subsystem_calls"]),
            str(report["scf_iterations"]),
            str(report["total_retries"]),
            f"{float(report['session_reuse_fraction']):.6g}",
            f"{float(report['electronic_step_time_seconds']):.6g}",
            f"{float(phases['abacus_overhead']):.6g}",
            f"{float(phases['session_startup']):.6g}",
        )))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument(
        "--labels", nargs="+",
        help="labels in the same order as the reports; defaults to parent names")
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    labels = arguments.labels or [path.parent.name for path in arguments.reports]
    try:
        summary = compare_reports(arguments.reports, labels)
    except (OSError, ValueError, json.JSONDecodeError,
            PerformanceSummaryError) as error:
        parser.error(str(error))
    if arguments.json_output is not None:
        arguments.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    sys.stdout.write(render_tsv(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
