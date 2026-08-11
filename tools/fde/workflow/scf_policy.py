"""Residual-, iteration-, and recovery-aware inner-SCF stage selection."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


def _finite_nonnegative(value: object) -> float | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0.0):
        return float(value)
    return None


def _baseline_index(stages: Sequence[Mapping[str, object]],
                    previous_density_rms: float | None) -> int:
    if previous_density_rms is None:
        return 0
    for index, stage in enumerate(stages):
        if previous_density_rms >= float(stage["minimum_density_rms"]):
            return index
    return len(stages) - 1


def _last_cycle_pressure(history: Sequence[Mapping[str, object]],
                         threshold: float) -> tuple[bool, bool, float]:
    if not history:
        return False, False, 0.0
    inner = history[-1].get("inner_scf", {})
    if not isinstance(inner, dict):
        return False, False, 0.0
    recovery = False
    maximum_fraction = 0.0
    for metrics in inner.values():
        if not isinstance(metrics, dict):
            continue
        recovery = recovery or int(metrics.get("retry_count", 0)) > 0
        iterations = _finite_nonnegative(metrics.get("iterations"))
        maximum = _finite_nonnegative(metrics.get("maximum_iterations"))
        if iterations is not None and maximum is not None and maximum > 0.0:
            maximum_fraction = max(maximum_fraction, iterations / maximum)
    return recovery, maximum_fraction >= threshold, maximum_fraction


def _stagnation_ratios(history: Sequence[Mapping[str, object]],
                       count: int) -> list[float]:
    residuals = [_finite_nonnegative(item.get("density_rms")) for item in history]
    residuals = [value for value in residuals if value is not None]
    if len(residuals) < count + 1:
        return []
    selected = residuals[-(count + 1):]
    ratios: list[float] = []
    for previous, current in zip(selected, selected[1:]):
        ratios.append(math.inf if previous == 0.0 else current / previous)
    return ratios


def select_adaptive_stage(
    stages: Sequence[Mapping[str, object]],
    cycle: int,
    previous_density_rms: float | None,
    maximum_cycles: int,
    strict_confirmations: int,
    force_strict_cycle: int,
    history: Sequence[Mapping[str, object]],
    auto_tune: Mapping[str, object],
) -> Dict[str, object]:
    """Return a stage plus an auditable explanation of the selection."""
    if not stages:
        raise ValueError("adaptive SCF requires at least one stage")
    baseline = _baseline_index(stages, previous_density_rms)
    selected = baseline
    reason = "initial_stage" if previous_density_rms is None else "residual_threshold"
    forced = cycle >= force_strict_cycle
    evidence: Dict[str, object] = {
        "baseline_stage": str(stages[baseline]["name"]),
        "previous_density_rms": previous_density_rms,
        "forced_strict": forced,
    }
    if forced:
        selected = len(stages) - 1
        reason = "strict_confirmation_budget"
    elif bool(auto_tune.get("enabled", False)) and selected < len(stages) - 1:
        pressure_threshold = float(
            auto_tune.get("iteration_pressure_fraction", 0.8))
        recovery, pressured, fraction = _last_cycle_pressure(
            history, pressure_threshold)
        stagnation_cycles = int(auto_tune.get("stagnation_cycles", 2))
        ratios = _stagnation_ratios(history, stagnation_cycles)
        stagnation_ratio = float(auto_tune.get("stagnation_ratio", 0.9))
        stagnating = (len(ratios) == stagnation_cycles
                      and all(ratio >= stagnation_ratio for ratio in ratios))
        evidence.update({
            "last_iteration_fraction": fraction,
            "recovery_used": recovery,
            "stagnation_ratios": ratios,
        })
        promote_on_recovery = bool(auto_tune.get("promote_on_recovery", True))
        if recovery and promote_on_recovery:
            selected += 1
            reason = "recovery_pressure"
        elif pressured:
            selected += 1
            reason = "iteration_pressure"
        elif stagnating:
            selected += 1
            reason = "outer_residual_stagnation"
    evidence["selected_stage"] = str(stages[selected]["name"])
    evidence["reason"] = reason
    evidence["remaining_cycles"] = maximum_cycles - cycle + 1
    evidence["required_strict_confirmations"] = strict_confirmations
    return {
        "stage": dict(stages[selected]),
        "stage_index": selected,
        "decision": evidence,
    }
