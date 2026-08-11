"""Nonoverlapping phase accounting for FDE subsystem and PES workflows."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


PHASE_NAMES = (
    "workflow_preparation",
    "session_startup",
    "electronic_steps",
    "abacus_overhead",
    "artifact_validation",
)


def _seconds(value: object) -> float:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0.0):
        return float(value)
    return 0.0


def subsystem_phase_profile(metrics: Mapping[str, object]) -> Dict[str, float]:
    """Build nonoverlapping phases from process/session and ABACUS metrics."""
    abacus_wall = _seconds(metrics.get("wall_time_seconds"))
    electronic = _seconds(metrics.get("electronic_step_time_seconds"))
    electronic_inside_wall = min(electronic, abacus_wall)
    return {
        "workflow_preparation": _seconds(
            metrics.get("workflow_preparation_seconds")),
        "session_startup": _seconds(metrics.get("session_startup_seconds")),
        "electronic_steps": electronic_inside_wall,
        "abacus_overhead": max(0.0, abacus_wall - electronic_inside_wall),
        "artifact_validation": _seconds(
            metrics.get("artifact_validation_seconds")),
    }


def aggregate_profiles(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    totals = {name: 0.0 for name in PHASE_NAMES}
    retries = 0
    reused = 0
    for record in records:
        profile = record.get("phase_times_seconds")
        if not isinstance(profile, dict):
            profile = subsystem_phase_profile(record)
        for name in PHASE_NAMES:
            totals[name] += _seconds(profile.get(name))
        retries += int(record.get("retry_count", 0))
        reused += int(bool(record.get("session_reused", False)))
    profiled = math.fsum(totals.values())
    abacus = totals["electronic_steps"] + totals["abacus_overhead"]
    return {
        "phase_totals_seconds": totals,
        "total_profiled_seconds": profiled,
        "electronic_step_fraction_of_abacus": (
            totals["electronic_steps"] / abacus if abacus > 0.0 else 0.0),
        "session_reuse_fraction": reused / len(records) if records else 0.0,
        "total_retries": retries,
    }
