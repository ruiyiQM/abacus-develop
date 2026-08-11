#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("prepare_example.py")
SPEC = importlib.util.spec_from_file_location("prepare_lih_folding", MODULE_PATH)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare)


class PrepareLihFoldingTest(unittest.TestCase):
    def test_generates_absolute_paths_and_population_exact_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for system in ("primitive", "supercell"):
                (root / system).mkdir(parents=True)
            for resource in (
                "resources/pseudopotentials/H_ONCV_PBE-1.0.upf",
                "resources/pseudopotentials/Li_ONCV_PBE-1.0.upf",
                "resources/orbitals/H_gga_8au_100Ry_2s1p.orb",
                "resources/orbitals/Li_gga_8au_100Ry_4s1p.orb",
            ):
                path = root / resource
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")

            prepare.prepare(root)

            primitive = (root / "primitive/FDE_CONFIG").read_text()
            supercell = (root / "supercell/FDE_CONFIG").read_text()
            self.assertIn(str((root / "primitive/li.fde_seed").resolve()), primitive)
            self.assertIn("KEDF pw91k", primitive)
            self.assertIn("FRAGMENT_XC pbe", primitive)
            self.assertIn("FRAGMENT Li 2 2 0 1", supercell)
            self.assertIn(
                "POPULATIONS 2 0",
                (root / "supercell/li.fde_seed").read_text(),
            )

    def test_rejects_missing_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "primitive").mkdir()
            (root / "supercell").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "missing pinned resource"):
                prepare.prepare(root)


if __name__ == "__main__":
    unittest.main()
