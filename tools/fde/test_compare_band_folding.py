import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("compare_band_folding.py")
SPEC = importlib.util.spec_from_file_location("compare_band_folding", MODULE_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def artifact(path: Path, bands, kpoints=(0.0,), state="fold"):
    lines = [
        "FDE_KPOINT_BANDS 1",
        f"STATE {state}",
        "FRAGMENT Li",
        "GEOMETRY test",
        "ORBITALS lih-dzp",
        f"AO_DIMENSION {len(bands[0])}",
        f"BAND_COUNT {len(bands[0])}",
        f"KPOINT_COUNT {2 * len(kpoints)}",
    ]
    for spin in (0, 1):
        for index, kpoint in enumerate(kpoints):
            values = bands[index]
            lines.append(
                f"KPOINT {spin} {index} {kpoint} 0 0 {1.0 / len(kpoints)}")
            lines.append(
                f"EIGENVALUES {len(values)} " + " ".join(str(value) for value in values))
    lines.append("END_FDE_KPOINT_BANDS")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class BandFoldingTest(unittest.TestCase):
    def test_matches_sorted_folded_spectrum_for_both_spins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact(root / "primitive", [[-2.0, 1.0], [-1.0, 2.0]], (0.0, 0.5))
            artifact(root / "supercell", [[-2.0, -1.0, 1.0, 2.0]])
            result = compare.compare_band_folding(
                compare.read_band_artifact(root / "primitive"),
                compare.read_band_artifact(root / "supercell"),
                1.0e-12)
            self.assertTrue(result["passed"])
            self.assertEqual(result["folding_factor"], 2)
            self.assertEqual(result["maximum_absolute_error_ry"], 0.0)

    def test_reports_failed_tolerance_without_hiding_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact(root / "primitive", [[-2.0], [-1.0]], (0.0, 0.5))
            artifact(root / "supercell", [[-2.0, -0.9]])
            result = compare.compare_band_folding(
                compare.read_band_artifact(root / "primitive"),
                compare.read_band_artifact(root / "supercell"),
                1.0e-3)
            self.assertFalse(result["passed"])
            self.assertAlmostEqual(result["maximum_absolute_error_ry"], 0.1)

    def test_rejects_mismatched_state_and_non_gamma_supercell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact(root / "primitive", [[-2.0], [-1.0]], (0.0, 0.5))
            artifact(root / "supercell", [[-2.0, -1.0]], state="other")
            with self.assertRaises(compare.BandFoldingError):
                compare.compare_band_folding(
                    compare.read_band_artifact(root / "primitive"),
                    compare.read_band_artifact(root / "supercell"),
                    1.0e-6)


if __name__ == "__main__":
    unittest.main()
