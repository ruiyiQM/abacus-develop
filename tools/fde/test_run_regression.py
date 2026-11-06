import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("run_regression.py")
SPEC = importlib.util.spec_from_file_location("run_fde_regression", MODULE_PATH)
regression = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(regression)


class DurableRegressionTest(unittest.TestCase):
    def test_manifest_includes_nested_helper_suites(self):
        manifest = json.loads((ROOT / "tools/fde/regression_manifest.json").read_text())
        starts = {suite["start_directory"] for suite in manifest["python_suites"]}
        self.assertIn("tools/fde/workflow", starts)
        self.assertIn("tools/fde/euler", starts)

    def test_python_suites_count_execution_and_reject_incomplete_results(self):
        cases = {
            "passing": ("import unittest\nclass Probe(unittest.TestCase):\n"
                        " def test_probe(self): self.assertEqual(2 + 2, 4)\n", True, 1, 0),
            "empty": ("", False, 0, 0),
            "skipped": ("import unittest\n@unittest.skip('missing dependency')\n"
                        "class Probe(unittest.TestCase):\n def test_probe(self): pass\n",
                        False, 0, 1),
            "import_error": ("import fde_nonexistent_dependency\n", False, 1, 0),
        }
        for label, (source, passed, executed, skipped) in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "test_probe.py").write_text(source, encoding="utf-8")
                result = regression.run_python_suites(
                    root, [{"label": label, "start_directory": "."}])[0]
                self.assertEqual(result["passed"], passed, result)
                self.assertEqual(result["executed"], executed, result)
                self.assertEqual(len(result["skipped"]), skipped, result)
                if skipped:
                    self.assertIn("missing dependency", result["skipped"][0]["reason"])

    def test_ctest_rejects_empty_disabled_skipped_or_unavailable_tests(self):
        if shutil.which("ctest") is None:
            self.skipTest("ctest executable is unavailable")
        cases = {
            "passing": ('add_test(MODULE_FDE_probe "' + sys.executable + '" "-c" "pass")\n', True),
            "empty": ("", False),
            "disabled": ('add_test(MODULE_FDE_probe "' + sys.executable + '" "-c" "pass")\n'
                         'set_tests_properties(MODULE_FDE_probe PROPERTIES DISABLED TRUE)\n', False),
            "skipped": ('add_test(MODULE_FDE_probe "' + sys.executable + '" "-c" "raise SystemExit(77)")\n'
                        'set_tests_properties(MODULE_FDE_probe PROPERTIES SKIP_RETURN_CODE 77)\n', False),
            "missing_executable": ('add_test(MODULE_FDE_probe "/fde-nonexistent-executable")\n', False),
        }
        for label, (source, passed) in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "CTestTestfile.cmake").write_text(source, encoding="utf-8")
                result = regression.run_native_ctest(root)
                self.assertEqual(result["passed"], passed, result)
                self.assertEqual(result["selected"], 0 if label == "empty" else 1, result)
                self.assertEqual(result["executed"], 1 if passed else 0, result)
                if label in ("disabled", "skipped", "missing_executable"):
                    self.assertTrue(result["incomplete"], result)

    def test_ctest_reports_conditionally_unregistered_native_tests(self):
        if shutil.which("ctest") is None:
            self.skipTest("ctest executable is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CTestTestfile.cmake").write_text(
                'add_test(MODULE_FDE_probe "' + sys.executable + '" "-c" "pass")\n')
            (root / "CMakeCache.txt").write_text(
                "ENABLE_JSON:BOOL=OFF\nENABLE_LIBXC:BOOL=ON\nENABLE_MPI:BOOL=OFF\n")
            result = regression.run_native_ctest(root)
            unavailable = {entry["name"]: entry["reason"]
                           for entry in result["unavailable_tests"]}
            self.assertIn("ENABLE_JSON=OFF", unavailable["MODULE_FDE_session_json"])
            self.assertIn("ENABLE_MPI=OFF", unavailable["MODULE_FDE_session_json_2rank"])
            self.assertIn("ENABLE_MPI=OFF", unavailable["MODULE_FDE_output_failure_4rank"])
            self.assertIn("ENABLE_MPI=OFF", unavailable["MODULE_FDE_native_workflow_periodic_hole_twist_uks_mpi_consistency"])
            self.assertIn("ENABLE_MPI=OFF", unavailable["MODULE_FDE_native_workflow_periodic_pair_twist_rks_mpi_consistency"])
            self.assertFalse(result["passed"])
            self.assertIn("MODULE_FDE_cell_volume", result["missing_tests"])
            self.assertIn("MODULE_FDE_native_workflow_periodic_hole_gamma_uks", result["missing_tests"])
            self.assertIn("MODULE_FDE_native_workflow_periodic_hole_twist_uks", result["missing_tests"])
            self.assertIn("MODULE_FDE_native_workflow_periodic_pair_twist_rks", result["missing_tests"])
            self.assertIn("MODULE_FDE_native_workflow_periodic_hole_boundary_consistency", result["missing_tests"])

    def test_ctest_preserves_native_workflow_metrics_and_rejects_missing_reports(self):
        if shutil.which("ctest") is None:
            self.skipTest("ctest executable is unavailable")
        for payload, passed in (("True", True), ("False", False), (None, False)):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                code = ("import json; print(json.dumps(dict(passed=" + payload + ", padding='x'*4096)))") if payload else "print('missing metrics')"
                (root / "CTestTestfile.cmake").write_text(
                    'add_test(MODULE_FDE_native_workflow_probe "' + sys.executable + '" "-c" "' + code + '")\n')
                result = regression.run_native_ctest(root)
                self.assertEqual(result["passed"], passed, result)
                if passed:
                    self.assertTrue(result["tests"][0]["metrics"]["passed"])
                    self.assertEqual(len(result["tests"][0]["metrics"]["padding"]), 4096)

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
