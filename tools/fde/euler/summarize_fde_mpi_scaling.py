#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


LAYOUTS = (
    "1rank_1thread",
    "4rank_1thread",
    "20rank_4thread",
    "40rank_4thread",
)
FINGERPRINTS = (
    "stru_sha256",
    "kpt_sha256",
    "fde_config_sha256",
    "source_commit",
    "binary_sha256",
)


def arguments():
    parser = argparse.ArgumentParser(
        description="Validate and summarize the Euler FDE MPI scaling run"
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--energy-tolerance-ry", type=float, default=1.0e-8)
    parser.add_argument("--drho-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def load_results(results_dir: Path):
    results = []
    for layout in LAYOUTS:
        path = results_dir / f"{layout}.json"
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if result.get("layout") != layout:
            raise RuntimeError(f"layout mismatch in {path}")
        results.append(result)
    return results


def validate_fingerprints(results):
    reference = results[0]
    for result in results[1:]:
        for name in FINGERPRINTS:
            if result.get(name) != reference.get(name):
                raise RuntimeError(
                    f"{name} differs between {reference['layout']} "
                    f"and {result['layout']}"
                )


def summarize(results, energy_tolerance_ry, drho_tolerance):
    reference = results[0]
    reference_energy = reference["etot_ry"]
    reference_drho = reference["drho"]
    reference_time = reference["scf_iteration_seconds"]
    four_rank_time = results[1]["scf_iteration_seconds"]
    cases = []
    for result in results:
        time = result["scf_iteration_seconds"]
        cores = result["total_cores"]
        delta_energy = result["etot_ry"] - reference_energy
        delta_drho = result["drho"] - reference_drho
        speedup = reference_time / time
        cases.append(
            {
                **result,
                "energy_delta_from_1rank_ry": delta_energy,
                "drho_delta_from_1rank": delta_drho,
                "speedup_from_1rank": speedup,
                "parallel_efficiency_from_1rank": speedup / cores,
                "speedup_from_4rank": four_rank_time / time,
            }
        )
    max_energy_delta = max(
        abs(case["energy_delta_from_1rank_ry"]) for case in cases
    )
    max_drho_delta = max(abs(case["drho_delta_from_1rank"]) for case in cases)
    passed = (
        max_energy_delta <= energy_tolerance_ry
        and max_drho_delta <= drho_tolerance
    )
    return {
        "schema": 1,
        "reference_layout": reference["layout"],
        "energy_tolerance_ry": energy_tolerance_ry,
        "drho_tolerance": drho_tolerance,
        "max_energy_delta_ry": max_energy_delta,
        "max_drho_delta": max_drho_delta,
        "consistent": passed,
        "cases": cases,
    }


def print_markdown(summary):
    print("| layout | nodes | ranks x threads | actual cores | SCF step (s) | "
          "energy delta (Ry) | speedup vs 1 rank | efficiency |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case in summary["cases"]:
        print(
            f"| {case['layout']} | {case['nodes']} | "
            f"{case['mpi_ranks']} x {case['omp_threads_per_rank']} | "
            f"{case['total_cores']} | {case['scf_iteration_seconds']:.2f} | "
            f"{case['energy_delta_from_1rank_ry']:.3e} | "
            f"{case['speedup_from_1rank']:.2f} | "
            f"{case['parallel_efficiency_from_1rank']:.3f} |"
        )
    status = "PASS" if summary["consistent"] else "FAIL"
    print(
        f"\nConsistency: **{status}**; maximum |dE| = "
        f"{summary['max_energy_delta_ry']:.3e} Ry, maximum |dDRHO| = "
        f"{summary['max_drho_delta']:.3e}."
    )


def main():
    args = arguments()
    if not math.isfinite(args.energy_tolerance_ry) or args.energy_tolerance_ry < 0:
        raise ValueError("energy tolerance must be finite and nonnegative")
    if not math.isfinite(args.drho_tolerance) or args.drho_tolerance < 0:
        raise ValueError("DRHO tolerance must be finite and nonnegative")
    results = load_results(args.results_dir)
    validate_fingerprints(results)
    summary = summarize(results, args.energy_tolerance_ry, args.drho_tolerance)
    if args.markdown:
        print_markdown(summary)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
