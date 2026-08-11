import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("prepare_example.py")
SPEC = importlib.util.spec_from_file_location("prepare_example", MODULE_PATH)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareExampleTest(unittest.TestCase):
    def test_reads_cube_grid_and_writes_compact_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cube = root / "grid.cube"
            cube.write_text(
                "title\ncomment\n0 0 0 0\n2 1 0 0\n1 0 2 0\n1 0 0 3\n",
                encoding="utf-8")
            grid = prepare.cube_grid(cube)
            self.assertEqual(grid[:3], (2, 1, 1))
            self.assertAlmostEqual(grid[3], 12.0)
            seed = root / "seed.fde"
            prepare.write_seed(seed, "g0", "reactant", "F", 4, 4, grid,
                               "0123456789abcdef", "revapbek")
            text = seed.read_text(encoding="utf-8")
            self.assertIn("FDE_UNIFORM_DENSITY_SEED 1", text)
            self.assertIn("POPULATIONS 4 4", text)
            self.assertIn("RHO_UNIFORM 0.33333333333333331", text)
            self.assertIn("sg15-v1.0-pbe-oncv-no-nlcc-0123456789abcdef", text)
            self.assertIn("standard-v2.0-dzp-100ry-0123456789abcdef", text)
            self.assertIn("FUNCTIONALS pbe revapbek", text)

    def test_renders_reaction_coordinate_without_changing_atom_order(self):
        template = (MODULE_PATH.parent / "STRU").read_text(encoding="utf-8")
        rendered = prepare.render_stru(template, 3.2, 1.8)
        self.assertIn("8.80000000 12.000000 12.000000", rendered)
        self.assertIn("13.80000000 12.000000 12.000000", rendered)
        self.assertLess(rendered.index("\nF\n"), rendered.index("\nC\n"))
        self.assertLess(rendered.index("\nC\n"), rendered.index("\nCl\n"))

    def test_grid_probe_input_has_anion_population_and_high_precision_cube(self):
        template = (MODULE_PATH.parent / "INPUT").read_text(encoding="utf-8")
        rendered = prepare.patch_input_text(template, {
            "nelec": 22, "nspin": 1, "nbands": 12, "out_chg": "2 10",
            "fde_task": "none",
        })
        self.assertIn("nelec                    22", rendered)
        self.assertIn("nspin                    1", rendered)
        self.assertIn("nbands                   12", rendered)
        self.assertIn("out_chg                  2 10", rendered)
        self.assertIn("fde_task                 none", rendered)
        self.assertEqual(rendered.count("out_chg"), 1)

    def test_benchmark_numeric_arguments_are_positive(self):
        self.assertEqual(prepare.positive_float("60"), 60.0)
        self.assertEqual(prepare.positive_integer("300"), 300)
        with self.assertRaises(prepare.argparse.ArgumentTypeError):
            prepare.positive_float("nan")
        with self.assertRaises(prepare.argparse.ArgumentTypeError):
            prepare.positive_integer("0")

    def test_default_resource_names_match_stru(self):
        manifest = prepare.load_resource_manifest()
        stru = (MODULE_PATH.parent / "STRU").read_text(encoding="utf-8")
        filenames = [Path(resource["path"]).name
                     for resource in manifest["resources"]]
        self.assertEqual(len(filenames), 8)
        for filename in filenames:
            self.assertIn(filename, stru)

    def test_example_is_one_explicit_uks_geometry(self):
        with (MODULE_PATH.parent / "geometry.csv").open(
                newline="", encoding="utf-8") as stream:
            geometries = list(csv.DictReader(stream))
        workflow = json.loads(
            (MODULE_PATH.parent / "workflow.template.json").read_text(encoding="utf-8"))
        self.assertEqual(len(geometries), 1)
        self.assertEqual(geometries[0]["label"], "uks_two_fermi")
        self.assertEqual(workflow["controls"]["spin_mode"], "uks")
        self.assertEqual(workflow["states"][0]["fragments"]["F"]["charge"], -1)
        self.assertEqual(workflow["states"][1]["fragments"]["F"]["spin"], 1)


if __name__ == "__main__":
    unittest.main()
