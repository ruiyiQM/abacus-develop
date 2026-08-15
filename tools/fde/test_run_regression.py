import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("run_regression.py")
SPEC = importlib.util.spec_from_file_location("run_fde_regression", MODULE_PATH)
regression = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(regression)


class DurableRegressionTest(unittest.TestCase):
    def test_current_manifest_and_compact_references_pass(self):
        report = regression.validate_manifest(
            ROOT, ROOT / "tools/fde/regression_manifest.json"
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["molecular_diabatic"]["maximum_delta"], 0.0)
        self.assertEqual(report["mpi_scaling"]["case_count"], 4)
        self.assertEqual(report["cuda_consistency"]["case_count"], 5)

    def test_band_folding_rejects_a_failed_or_over_budget_reference(self):
        reference = json.loads((
            ROOT / "examples/fdedft/02_lih_kpoint_folding/reference/"
            "band_folding_result.json.ref"
        ).read_text(encoding="utf-8"))
        reference["maximum_absolute_error_ry"] = 2.0 * reference["tolerance_ry"]
        with self.assertRaisesRegex(regression.RegressionError, "folding reference fails"):
            regression.validate_band_folding_reference(reference)

    def test_cuda_reference_rejects_a_component_mismatch(self):
        reference = json.loads((
            ROOT / "tools/fde/euler/reference/fde_cuda_euler_2026-08-10.json"
        ).read_text(encoding="utf-8"))
        reference["cases"][0]["absolute_nake_delta_ry"] = 1.0
        with self.assertRaisesRegex(regression.RegressionError, "agreement budget"):
            regression.validate_cuda_reference(reference)

    def test_reference_only_cli_writes_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                returncode = regression.main([
                    "--source-root", str(ROOT),
                    "--skip-python-tests",
                    "--output", str(output),
                ])
            self.assertEqual(returncode, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["python_suites"], [])
            self.assertIsNone(report["native_ctest"])


if __name__ == "__main__":
    unittest.main()
