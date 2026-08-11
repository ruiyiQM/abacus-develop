#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("compare_scientific_runs.py")
SPEC = importlib.util.spec_from_file_location("compare_scientific_runs", MODULE_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def write_pes(path: Path, points):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "geometry",
                "coordinate_angstrom",
                "reactant_energy_ry",
                "product_energy_ry",
                "overlap",
                "h12_ry",
                "orthogonalized_coupling_ry",
            ]
        )
        writer.writerows(points)


def budget():
    return {
        "schema_version": 1,
        "case": "unit-test",
        "coordinate_tolerance_angstrom": 1e-10,
        "require_same_crossing_brackets": True,
        "tolerances": {
            "maximum_state_energy_delta_ry": 1e-4,
            "maximum_energy_gap_delta_ry": 1e-4,
            "maximum_overlap_delta": 1e-5,
            "maximum_h12_delta_ry": 1e-4,
            "maximum_coupling_delta_ry": 1e-5,
        },
        "convergence": {
            "required": True,
            "maximum_density_rms": 1e-5,
            "maximum_cycles": 20,
        },
    }


class ScientificComparisonTest(unittest.TestCase):
    def test_phase_invariant_budget_accepts_determinant_sign_flip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.tsv"
            candidate = root / "candidate.tsv"
            write_pes(
                baseline,
                [["g0", 0.0, -10.0, -9.9, 0.01, -0.1, -0.002]],
            )
            write_pes(
                candidate,
                [["g0", 0.0, -10.0, -9.9, -0.01, 0.1, 0.002]],
            )
            settings = budget()
            settings["phase_invariant_off_diagonal"] = True
            settings["convergence"]["required"] = False

            report = comparison.compare(settings, baseline, candidate)

            self.assertTrue(report["passed"])
            self.assertEqual(
                report["off_diagonal_comparison"],
                "determinant_phase_aligned",
            )
            self.assertEqual(report["maxima"]["coupling_delta_ry"], 0.0)
            self.assertEqual(
                report["points"][0]["determinant_phase_alignment"], -1.0)
            self.assertAlmostEqual(
                report["points"][0]["signed_off_diagonal_deltas"]
                ["coupling_delta_ry"],
                0.004,
            )

    def test_phase_alignment_rejects_inconsistent_h12_sign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.tsv"
            candidate = root / "candidate.tsv"
            write_pes(
                baseline,
                [["g0", 0.0, -10.0, -9.9, 0.01, -0.1, -0.002]],
            )
            write_pes(
                candidate,
                [["g0", 0.0, -10.0, -9.9, -0.01, -0.1, 0.002]],
            )
            settings = budget()
            settings["phase_invariant_off_diagonal"] = True
            settings["convergence"]["required"] = False

            report = comparison.compare(settings, baseline, candidate)

            self.assertFalse(report["passed"])
            self.assertAlmostEqual(report["maxima"]["h12_delta_ry"], 0.2)

    def test_signed_budget_still_rejects_determinant_sign_flip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.tsv"
            candidate = root / "candidate.tsv"
            write_pes(
                baseline,
                [["g0", 0.0, -10.0, -9.9, 0.01, -0.1, -0.002]],
            )
            write_pes(
                candidate,
                [["g0", 0.0, -10.0, -9.9, -0.01, 0.1, 0.002]],
            )
            settings = budget()
            settings["convergence"]["required"] = False

            report = comparison.compare(settings, baseline, candidate)

            self.assertFalse(report["passed"])
            self.assertEqual(report["off_diagonal_comparison"], "signed")

    def test_passes_small_numerical_changes_and_tracks_crossing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.tsv"
            candidate = root / "candidate.tsv"
            convergence = root / "convergence.json"
            write_pes(
                baseline,
                [
                    ["g0", -1.0, -10.0, -9.8, 0.01, -0.1, 0.002],
                    ["g1", 1.0, -9.8, -10.0, 0.01, -0.1, 0.002],
                ],
            )
            write_pes(
                candidate,
                [
                    ["x0", -1.0, -9.99999, -9.79999, 0.010001, -0.1, 0.002001],
                    ["x1", 1.0, -9.79999, -9.99999, 0.010001, -0.1, 0.002001],
                ],
            )
            convergence.write_text(
                json.dumps(
                    {
                        "reactant": {
                            "converged": True,
                            "cycle": 5,
                            "density_rms": 1e-6,
                        },
                        "product": {
                            "converged": True,
                            "cycle": 6,
                            "density_rms": 2e-6,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = comparison.compare(
                budget(), baseline, candidate, convergence
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["crossing_brackets"]["candidate"], [[-1.0, 1.0]])

    def test_fails_coupling_budget_and_missing_convergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.tsv"
            candidate = root / "candidate.tsv"
            write_pes(baseline, [["g0", 0.0, -10.0, -9.9, 0.01, -0.1, 0.002]])
            write_pes(candidate, [["g0", 0.0, -10.0, -9.9, 0.01, -0.1, 0.003]])
            report = comparison.compare(budget(), baseline, candidate)
            self.assertFalse(report["passed"])
            self.assertTrue(any("coupling_delta" in item for item in report["failures"]))
            self.assertIn(
                "required convergence data were not provided", report["failures"]
            )

    def test_rejects_changed_coordinate_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.tsv"
            candidate = root / "candidate.tsv"
            write_pes(baseline, [["g0", 0.0, -10.0, -9.9, 0.01, -0.1, 0.002]])
            write_pes(candidate, [["g1", 1.0, -10.0, -9.9, 0.01, -0.1, 0.002]])
            with self.assertRaises(comparison.ScientificComparisonError):
                comparison.compare(budget(), baseline, candidate)

    def test_reads_workflow_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "g0" / "reactant"
            state.mkdir(parents=True)
            (state / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "state": "reactant",
                        "converged": True,
                        "cycle": 4,
                        "density_rms": 1e-7,
                    }
                ),
                encoding="utf-8",
            )
            records = comparison.load_convergence(root)
            self.assertEqual(records[0]["state"], "reactant")


if __name__ == "__main__":
    unittest.main()
