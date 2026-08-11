#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("summarize_performance.py")
SPEC = importlib.util.spec_from_file_location("summarize_performance", MODULE_PATH)
performance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(performance)


class SummarizePerformanceTest(unittest.TestCase):
    def write_report(self, path: Path, wall: float, schema: int = 2) -> None:
        payload = {
            "schema_version": schema,
            "total_wall_time_seconds": wall,
            "total_electronic_step_time_seconds": 0.4 * wall,
            "total_subsystem_calls": 4,
            "total_scf_iterations": 20,
        }
        if schema == 2:
            payload.update({
                "total_retries": 1,
                "session_reuse_fraction": 0.75,
                "total_profiled_seconds": wall,
                "phase_totals_seconds": {
                    "workflow_preparation": 0.02 * wall,
                    "session_startup": 0.03 * wall,
                    "electronic_steps": 0.4 * wall,
                    "abacus_overhead": 0.54 * wall,
                    "artifact_validation": 0.01 * wall,
                },
            })
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_reports_speedup_and_structured_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            self.write_report(baseline, 100.0)
            self.write_report(candidate, 50.0)

            summary = performance.compare_reports(
                [baseline, candidate], ["cpu", "gpu"])

            self.assertEqual(summary["baseline"], "cpu")
            self.assertEqual(
                summary["reports"][1]["speedup_vs_baseline"], 2.0)
            self.assertEqual(
                summary["reports"][0]["profiled_time_seconds"], 100.0)
            self.assertIn("gpu\t50\t50\t2", performance.render_tsv(summary))

    def test_accepts_legacy_schema_without_inventing_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "legacy.json"
            self.write_report(report, 10.0, schema=1)

            summary = performance.compare_reports([report], ["legacy"])

            item = summary["reports"][0]
            self.assertEqual(item["profiled_time_seconds"], 0.0)
            self.assertEqual(item["session_reuse_fraction"], 0.0)
            self.assertEqual(
                item["comparison_time_source"], "total_wall_time_seconds")

    def test_rejects_inconsistent_profile_total(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "invalid-profile.json"
            self.write_report(report, 10.0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["total_profiled_seconds"] = 9.0
            report.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(performance.PerformanceSummaryError):
                performance.compare_reports([report], ["invalid"])

    def test_rejects_invalid_reuse_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "invalid.json"
            self.write_report(report, 10.0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["session_reuse_fraction"] = 1.5
            report.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(performance.PerformanceSummaryError):
                performance.compare_reports([report], ["invalid"])


if __name__ == "__main__":
    unittest.main()
