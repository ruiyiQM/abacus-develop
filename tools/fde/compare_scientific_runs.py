#!/usr/bin/env python3
"""Compare two FDE PES data sets against an explicit scientific error budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Mapping, Sequence, Tuple


class ScientificComparisonError(RuntimeError):
    pass


def read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not header or not rows:
        raise ScientificComparisonError(f"empty TSV data set: {path}")
    return header, rows


def finite(row: Mapping[str, str], column: str, source: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ScientificComparisonError(
            f"invalid {column!r} value in {source}"
        ) from error
    if not math.isfinite(value):
        raise ScientificComparisonError(
            f"nonfinite {column!r} value in {source}"
        )
    return value


def state_energy_columns(header: Sequence[str]) -> List[str]:
    excluded = {"h12_ry", "orthogonalized_coupling_ry"}
    columns = [
        name for name in header
        if name.endswith("_energy_ry") and name not in excluded
    ]
    if len(columns) != 2:
        raise ScientificComparisonError(
            "scientific comparison currently requires exactly two state-energy columns"
        )
    return columns


def indexed_rows(
    path: Path,
    coordinate_tolerance: float,
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    header, rows = read_tsv(path)
    required = {
        "geometry",
        "coordinate_angstrom",
        "overlap",
        "h12_ry",
        "orthogonalized_coupling_ry",
    }
    missing = required.difference(header)
    if missing:
        raise ScientificComparisonError(
            f"{path} lacks required columns: {', '.join(sorted(missing))}"
        )
    energies = state_energy_columns(header)
    indexed: Dict[str, Dict[str, str]] = {}
    for row in rows:
        label = row.get("geometry", "")
        if not label or label in indexed:
            raise ScientificComparisonError(
                f"geometry labels must be nonempty and unique in {path}"
            )
        coordinate = finite(row, "coordinate_angstrom", path)
        if coordinate_tolerance > 0.0:
            key = f"{round(coordinate / coordinate_tolerance):d}"
        else:
            key = f"{coordinate:.17g}"
        if key in indexed:
            raise ScientificComparisonError(
                f"coordinates are not unique within tolerance in {path}"
            )
        indexed[key] = row
    return energies, indexed


def crossing_brackets(
    rows: Sequence[Mapping[str, str]],
    energies: Sequence[str],
    source: Path,
) -> List[List[float]]:
    ordered = sorted(
        rows, key=lambda row: finite(row, "coordinate_angstrom", source)
    )
    brackets: List[List[float]] = []
    previous_coordinate = finite(ordered[0], "coordinate_angstrom", source)
    previous_gap = (
        finite(ordered[0], energies[1], source)
        - finite(ordered[0], energies[0], source)
    )
    for row in ordered[1:]:
        coordinate = finite(row, "coordinate_angstrom", source)
        gap = finite(row, energies[1], source) - finite(row, energies[0], source)
        if previous_gap == 0.0:
            brackets.append([previous_coordinate, previous_coordinate])
        elif gap == 0.0 or previous_gap * gap < 0.0:
            brackets.append([previous_coordinate, coordinate])
        previous_coordinate = coordinate
        previous_gap = gap
    if previous_gap == 0.0 and (
        not brackets or brackets[-1][1] != previous_coordinate
    ):
        brackets.append([previous_coordinate, previous_coordinate])
    return brackets


def load_convergence(path: Path) -> List[Dict[str, object]]:
    if path.is_dir():
        records = []
        for checkpoint_path in sorted(path.glob("*/*/checkpoint.json")):
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(checkpoint, dict):
                raise ScientificComparisonError(
                    f"checkpoint must be an object: {checkpoint_path}"
                )
            records.append(checkpoint)
        if not records:
            raise ScientificComparisonError(
                f"no geometry/state checkpoints found below {path}"
            )
        return records
    if path.suffix == ".json" or path.name.endswith(".json.ref"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ScientificComparisonError("convergence JSON must be an object")
        records = []
        for state, values in payload.items():
            if not isinstance(values, dict):
                raise ScientificComparisonError(
                    f"convergence state {state!r} must be an object"
                )
            records.append({"state": state, **values})
        return records
    _, rows = read_tsv(path)
    return [dict(row) for row in rows]


def convergence_report(
    path: Path | None,
    budget: Mapping[str, object],
) -> Dict[str, object]:
    if path is None:
        return {"provided": False, "passed": not budget.get("required", False)}
    records = load_convergence(path)
    maximum_density = float(budget.get("maximum_density_rms", math.inf))
    maximum_cycles = int(budget.get("maximum_cycles", 2**31 - 1))
    failures: List[str] = []
    normalized = []
    for record in records:
        state = str(record.get("state", "unknown"))
        converged_value = record.get("converged", False)
        converged = (
            converged_value is True
            or str(converged_value).strip().lower() in ("1", "true", "yes")
        )
        try:
            density = float(record.get("density_rms", "nan"))
            cycle = int(record.get("cycle", -1))
        except (TypeError, ValueError) as error:
            raise ScientificComparisonError(
                f"invalid convergence record for state {state}"
            ) from error
        if not converged:
            failures.append(f"{state}: not converged")
        if not math.isfinite(density) or density > maximum_density:
            failures.append(
                f"{state}: density RMS {density:.6g} exceeds {maximum_density:.6g}"
            )
        if cycle < 0 or cycle > maximum_cycles:
            failures.append(
                f"{state}: cycle {cycle} exceeds {maximum_cycles}"
            )
        normalized.append(
            {
                "state": state,
                "converged": converged,
                "cycle": cycle,
                "density_rms": density,
            }
        )
    return {
        "provided": True,
        "passed": not failures,
        "limits": {
            "maximum_density_rms": maximum_density,
            "maximum_cycles": maximum_cycles,
        },
        "records": normalized,
        "failures": failures,
    }


def compare(
    budget: Mapping[str, object],
    baseline_path: Path,
    candidate_path: Path,
    convergence_path: Path | None = None,
) -> Dict[str, object]:
    if int(budget.get("schema_version", 0)) != 1:
        raise ScientificComparisonError("unsupported scientific budget schema")
    coordinate_tolerance = float(budget.get("coordinate_tolerance_angstrom", 1e-10))
    if not math.isfinite(coordinate_tolerance) or coordinate_tolerance < 0.0:
        raise ScientificComparisonError("coordinate tolerance must be nonnegative")
    baseline_energies, baseline = indexed_rows(
        baseline_path, coordinate_tolerance
    )
    candidate_energies, candidate = indexed_rows(
        candidate_path, coordinate_tolerance
    )
    if baseline_energies != candidate_energies:
        raise ScientificComparisonError("state-energy columns differ between runs")
    if set(baseline) != set(candidate):
        raise ScientificComparisonError("coordinate sets differ between runs")
    phase_invariant = budget.get("phase_invariant_off_diagonal", False)
    if not isinstance(phase_invariant, bool):
        raise ScientificComparisonError(
            "phase_invariant_off_diagonal must be a Boolean")

    maxima = {
        "state_energy_delta_ry": 0.0,
        "energy_gap_delta_ry": 0.0,
        "overlap_delta": 0.0,
        "h12_delta_ry": 0.0,
        "coupling_delta_ry": 0.0,
    }
    points = []
    for key in sorted(baseline, key=lambda item: float(item)):
        reference = baseline[key]
        trial = candidate[key]
        reference_energies = [
            finite(reference, column, baseline_path)
            for column in baseline_energies
        ]
        trial_energies = [
            finite(trial, column, candidate_path)
            for column in candidate_energies
        ]
        energy_deltas = [
            trial_value - reference_value
            for trial_value, reference_value in zip(
                trial_energies, reference_energies
            )
        ]
        reference_gap = reference_energies[1] - reference_energies[0]
        trial_gap = trial_energies[1] - trial_energies[0]
        off_diagonal_columns = (
            ("overlap", "overlap_delta"),
            ("h12_ry", "h12_delta_ry"),
            ("orthogonalized_coupling_ry", "coupling_delta_ry"),
        )
        reference_off_diagonal = {
            column: finite(reference, column, baseline_path)
            for column, _ in off_diagonal_columns
        }
        trial_off_diagonal = {
            column: finite(trial, column, candidate_path)
            for column, _ in off_diagonal_columns
        }
        determinant_phase = 1.0
        if phase_invariant:
            for column, _ in off_diagonal_columns:
                reference_value = reference_off_diagonal[column]
                trial_value = trial_off_diagonal[column]
                if abs(reference_value) > 1.0e-15 and abs(trial_value) > 1.0e-15:
                    determinant_phase = (
                        1.0 if reference_value * trial_value >= 0.0 else -1.0)
                    break
        off_diagonal_deltas = {}
        signed_off_diagonal_deltas = {}
        for column, metric in off_diagonal_columns:
            reference_value = reference_off_diagonal[column]
            trial_value = trial_off_diagonal[column]
            signed_off_diagonal_deltas[metric] = trial_value - reference_value
            off_diagonal_deltas[metric] = (
                determinant_phase * trial_value - reference_value)
        maxima["state_energy_delta_ry"] = max(
            maxima["state_energy_delta_ry"],
            *(abs(value) for value in energy_deltas),
        )
        maxima["energy_gap_delta_ry"] = max(
            maxima["energy_gap_delta_ry"], abs(trial_gap - reference_gap)
        )
        maxima["overlap_delta"] = max(
            maxima["overlap_delta"],
            abs(off_diagonal_deltas["overlap_delta"]),
        )
        maxima["h12_delta_ry"] = max(
            maxima["h12_delta_ry"],
            abs(off_diagonal_deltas["h12_delta_ry"]),
        )
        maxima["coupling_delta_ry"] = max(
            maxima["coupling_delta_ry"],
            abs(off_diagonal_deltas["coupling_delta_ry"]),
        )
        point = {
            "geometry": trial["geometry"],
            "coordinate_angstrom": finite(
                trial, "coordinate_angstrom", candidate_path
            ),
            "state_energy_deltas_ry": dict(
                zip(candidate_energies, energy_deltas)
            ),
            "energy_gap_delta_ry": trial_gap - reference_gap,
            **off_diagonal_deltas,
        }
        if phase_invariant:
            point["determinant_phase_alignment"] = determinant_phase
            point["signed_off_diagonal_deltas"] = signed_off_diagonal_deltas
        points.append(point)

    tolerances = budget.get("tolerances", {})
    if not isinstance(tolerances, dict):
        raise ScientificComparisonError("budget tolerances must be an object")
    failures = []
    for metric, maximum in maxima.items():
        tolerance_name = f"maximum_{metric}"
        try:
            tolerance = float(tolerances[tolerance_name])
        except (KeyError, TypeError, ValueError) as error:
            raise ScientificComparisonError(
                f"missing or invalid tolerance {tolerance_name}"
            ) from error
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ScientificComparisonError(
                f"tolerance {tolerance_name} must be finite and nonnegative"
            )
        if maximum > tolerance:
            failures.append(
                f"{metric} {maximum:.6g} exceeds {tolerance:.6g}"
            )

    baseline_brackets = crossing_brackets(
        list(baseline.values()), baseline_energies, baseline_path
    )
    candidate_brackets = crossing_brackets(
        list(candidate.values()), candidate_energies, candidate_path
    )
    if budget.get("require_same_crossing_brackets", True):
        if baseline_brackets != candidate_brackets:
            failures.append("crossing brackets differ between runs")

    convergence_budget = budget.get("convergence", {})
    if not isinstance(convergence_budget, dict):
        raise ScientificComparisonError("convergence budget must be an object")
    convergence = convergence_report(convergence_path, convergence_budget)
    failures.extend(convergence.get("failures", []))
    if not convergence["passed"] and not convergence.get("failures"):
        failures.append("required convergence data were not provided")

    return {
        "schema_version": 1,
        "case": budget.get("case", "unknown"),
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "state_energy_columns": baseline_energies,
        "point_count": len(points),
        "off_diagonal_comparison": (
            "determinant_phase_aligned" if phase_invariant else "signed"),
        "maxima": maxima,
        "crossing_brackets": {
            "baseline": baseline_brackets,
            "candidate": candidate_brackets,
        },
        "convergence": convergence,
        "points": points,
        "failures": failures,
        "passed": not failures,
    }


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budget", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--candidate-convergence", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    budget = json.loads(args.budget.read_text(encoding="utf-8"))
    report = compare(
        budget,
        args.baseline,
        args.candidate,
        args.candidate_convergence,
    )
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
