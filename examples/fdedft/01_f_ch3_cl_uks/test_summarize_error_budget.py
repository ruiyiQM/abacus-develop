#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("summarize_error_budget.py")
SPEC = importlib.util.spec_from_file_location("summarize_error_budget", MODULE_PATH)
summary_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary_module)
SOURCE_ROOT = MODULE_PATH.resolve().parents[3]
REFERENCE = MODULE_PATH.parent / "reference/fde_pes.tsv.ref"


class ErrorBudgetSummaryTest(unittest.TestCase):
    def test_identical_variants_pass_required_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark_root = Path(directory)
            for variant in ("production", "tight_scf", "cutoff60"):
                work = benchmark_root / variant / f"work-{variant}"
                work.mkdir(parents=True)
                shutil.copy2(REFERENCE, work / "fde_pes.tsv")
                for state, cycle in (("reactant", 5), ("product", 6)):
                    state_directory = work / "uks_two_fermi" / state
                    state_directory.mkdir(parents=True)
                    (state_directory / "checkpoint.json").write_text(
                        json.dumps(
                            {
                                "state": state,
                                "converged": True,
                                "cycle": cycle,
                                "density_rms": 1e-6,
                            }
                        ),
                        encoding="utf-8",
                    )

            report = summary_module.summarize(SOURCE_ROOT, benchmark_root)

            self.assertTrue(report["passed"])
            self.assertEqual(len(report["comparisons"]), 3)
            self.assertFalse(report["comparisons"][2]["required"])


if __name__ == "__main__":
    unittest.main()
