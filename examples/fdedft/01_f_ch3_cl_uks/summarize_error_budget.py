#!/usr/bin/env python3
"""Summarize the production, tight-SCF, and 60 Ry FDE benchmark variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List


def load_comparison_tool(source_root: Path):
    path = source_root / "tools/fde/compare_scientific_runs.py"
    specification = importlib.util.spec_from_file_location(
        "compare_scientific_runs", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import scientific comparison tool: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def summarize(source_root: Path, benchmark_root: Path) -> Dict[str, object]:
    comparison = load_comparison_tool(source_root)
    case_root = source_root / "examples/fdedft/01_f_ch3_cl_uks"
    budget = json.loads(
        (case_root / "scientific_error_budget.json").read_text(encoding="utf-8")
    )
    variants = {
        name: benchmark_root / name / f"work-{name}"
        for name in ("production", "tight_scf", "cutoff60")
    }
    definitions = (
        (
            "committed_reference_to_production",
            case_root / "reference/fde_pes.tsv.ref",
            variants["production"] / "fde_pes.tsv",
            variants["production"],
            True,
        ),
        (
            "production_to_tight_scf",
            variants["production"] / "fde_pes.tsv",
            variants["tight_scf"] / "fde_pes.tsv",
            variants["tight_scf"],
            True,
        ),
        (
            "tight_scf_to_60ry",
            variants["tight_scf"] / "fde_pes.tsv",
            variants["cutoff60"] / "fde_pes.tsv",
            variants["cutoff60"],
            False,
        ),
    )
    reports: List[Dict[str, object]] = []
    required_failures = []
    for label, baseline, candidate, convergence, required in definitions:
        report = comparison.compare(
            budget, baseline, candidate, convergence
        )
        report["label"] = label
        report["required"] = required
        reports.append(report)
        if required and not report["passed"]:
            required_failures.append(label)
    return {
        "schema_version": 1,
        "case": budget["case"],
        "benchmark_root": str(benchmark_root.resolve()),
        "passed": not required_failures,
        "required_failures": required_failures,
        "comparisons": reports,
        "interpretation": {
            "production": "Reproduces the committed 40 Ry/DZP calculation.",
            "tight_scf": "Measures inner/outer convergence error at the same grid.",
            "cutoff60": "Reports grid sensitivity and is informative until a converged cutoff budget is established."
        },
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    source_root = args.source_root.resolve(strict=True)
    benchmark_root = args.benchmark_root.resolve(strict=True)
    report = summarize(source_root, benchmark_root)
    output = (
        args.output.resolve()
        if args.output is not None
        else benchmark_root / "scientific_error_budget_report.json"
    )
    comparison = load_comparison_tool(source_root)
    comparison.write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
