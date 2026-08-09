import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("fde_workflow.py")
SPEC = importlib.util.spec_from_file_location("fde_workflow", MODULE_PATH)
fde_workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fde_workflow)


FAKE_ABACUS = r'''#!/usr/bin/env python3
import os
from pathlib import Path


def value(path, key):
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].lower() == key.lower():
            return fields[1]
    raise RuntimeError("missing " + key)


cwd = Path.cwd()
task = value(cwd / "INPUT", "fde_task")
print("OMP_NUM_THREADS=" + os.environ.get("OMP_NUM_THREADS", ""))
if task == "diabatic_postprocess":
    config = Path(value(cwd / "INPUT", "fde_config"))
    text = config.read_text(encoding="utf-8")
    if text.count("DETERMINANT ") != 2 or text.count("LINEARIZED_STATE ") != 2:
        raise RuntimeError("postprocess config is incomplete")
    (cwd / "fde_diabatic.fde_diabatic").write_text(
        "FDE_DIABATIC_RESULT 1\n"
        "PAIR 0 1 0.8 -22.7 0.25\n"
        "ADIABATIC_ENERGIES_RY 2 -23.5 -22.3\nEND\n",
        encoding="utf-8")
    raise SystemExit(0)

config = Path(value(cwd / "INPUT", "fde_config"))
state = value(config, "ACTIVE_STATE")
fragment = value(config, "ACTIVE_FRAGMENT")
cycle = int(cwd.parent.name.split("-")[1])
output = cwd / "OUT.fake"
output.mkdir()
(output / "fake-CHARGE-DENSITY.restart").write_text("restart", encoding="utf-8")
populations = {
    ("reactant", "F"): (4, 4), ("reactant", "CH3Cl"): (7, 7),
    ("product", "F"): (4, 3), ("product", "CH3Cl"): (7, 8),
}
alpha, beta = populations[(state, fragment)]
density = cwd / "result.fde_density"
density.write_text(
    "FDE_DENSITY_ARTIFACT 1\n"
    f"FRAGMENT {fragment}\nSTATE {state}\nGEOMETRY g0\n"
    "GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS fake\nORBITALS fake\n"
    "CORE_DENSITY none\nFUNCTIONALS pbe lc94\nGRID 1 1 1 1\n"
    f"POPULATIONS {alpha} {beta}\nSCF {cycle} 1\nENERGIES_RY 0 0\n"
    f"RHO_ALPHA 1 {alpha}\nRHO_BETA 1 {beta}\nEND\n",
    encoding="utf-8")

identity = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
hamiltonian = "-2 0 0 0 0 -1 0 0 0 0 1 0 0 0 0 2"
if fragment == "F":
    active = 0
    alpha_orbital = "0.8 0 0.6 0" if state == "product" else "1 0 0 0"
    beta_orbital = "0 0 1 0"
    subsystem = -10.0 if state == "reactant" else -9.5
else:
    active = 1
    alpha_orbital = "0 1 0 0"
    beta_orbital = "0 0 0 1"
    subsystem = -12.0 if state == "reactant" else -11.5
fragment_artifact = cwd / "result.fde_fragment"
fragment_artifact.write_text(
    "FDE_FRAGMENT_SCF_ARTIFACT 1\n"
    f"STATE {state}\nFRAGMENT {fragment}\nGEOMETRY g0\nCYCLE {cycle}\n"
    "ORBITAL_BASIS fake\nDENSITY_PATH result.fde_density\nAO_DIMENSION 4\n"
    f"ACTIVE_ORBITALS 1 {active}\nSUBSYSTEM_TOTAL_ENERGY_RY {subsystem}\n"
    "ION_ION_ENERGY_RY 2\nHARTREE_CROSS_ENERGY_RY 0.5\n"
    "NONADDITIVE_KINETIC_ENERGY_RY 0.3\nNONADDITIVE_XC_ENERGY_RY -0.2\n"
    f"ALPHA 1\nORBITAL {fragment} -1 {alpha_orbital}\n"
    f"BETA 1\nORBITAL {fragment} -0.5 {beta_orbital}\n"
    f"AO_OVERLAP 16 {identity}\nHAMILTONIAN_ALPHA_RY 16 {hamiltonian}\n"
    f"HAMILTONIAN_BETA_RY 16 {hamiltonian}\nEND\n",
    encoding="utf-8")
'''


def write_seed(path: Path, state: str, fragment: str, alpha: int, beta: int) -> None:
    path.write_text(
        "FDE_UNIFORM_DENSITY_SEED 1\n"
        f"FRAGMENT {fragment}\nSTATE {state}\nGEOMETRY g0\n"
        "GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS fake\nORBITALS fake\n"
        "CORE_DENSITY none\nFUNCTIONALS pbe lc94\nGRID 1 1 1 1\n"
        f"POPULATIONS {alpha} {beta}\nRHO_UNIFORM {alpha} {beta}\nEND\n",
        encoding="utf-8")


class FdeWorkflowEndToEndTest(unittest.TestCase):
    def test_two_states_freeze_thaw_postprocess_and_pes_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_abacus = root / "fake_abacus.py"
            fake_abacus.write_text(FAKE_ABACUS, encoding="utf-8")
            template = root / "template"
            template.mkdir()
            (template / "INPUT").write_text("INPUT_PARAMETERS\n", encoding="utf-8")
            seeds = root / "seeds"
            seeds.mkdir()
            populations = {
                ("reactant", "F"): (4, 4), ("reactant", "CH3Cl"): (7, 7),
                ("product", "F"): (4, 3), ("product", "CH3Cl"): (7, 8),
            }
            initial = {"reactant": {}, "product": {}}
            for (state, fragment), (alpha, beta) in populations.items():
                seed = seeds / f"{state}_{fragment}.fde_seed"
                write_seed(seed, state, fragment, alpha, beta)
                initial[state][fragment] = str(seed)

            work = root / "work"
            specification = {
                "schema_version": 1,
                "abacus_command": [sys.executable, str(fake_abacus)],
                "postprocess_command": [sys.executable, str(fake_abacus)],
                "run_postprocess": True,
                "work_directory": str(work),
                "fragments": [
                    {"label": "F", "neutral_valence_electrons": 7,
                     "atom_indices": [0]},
                    {"label": "CH3Cl", "neutral_valence_electrons": 14,
                     "atom_indices": [1, 2, 3, 4, 5]},
                ],
                "states": [
                    {"label": "reactant", "total_charge": -1, "total_spin": 0,
                     "fragments": {"F": {"charge": -1, "spin": 0},
                                   "CH3Cl": {"charge": 0, "spin": 0}}},
                    {"label": "product", "total_charge": -1, "total_spin": 0,
                     "fragments": {"F": {"charge": 0, "spin": 1},
                                   "CH3Cl": {"charge": -1, "spin": -1}}},
                ],
                "controls": {"maximum_freeze_thaw_cycles": 3,
                             "freeze_thaw_density_tolerance": 1e-12,
                             "energy_tolerance_ry": 1e-12,
                             "ks_solver": "genelpa",
                             "kpar": 1,
                             "retain_completed_cycles": 1,
                             "remove_abacus_restart_files": True,
                             "update_order": ["F", "CH3Cl"]},
                "geometries": [{"label": "g0", "coordinate_angstrom": 0.2,
                                "template_directory": str(template),
                                "initial_densities": initial}],
            }
            spec_path = root / "workflow.json"
            spec_path.write_text(json.dumps(specification), encoding="utf-8")

            fde_workflow.run_workflow(spec_path)

            pes = json.loads((work / "fde_pes.json").read_text(encoding="utf-8"))
            point = pes["points"][0]
            self.assertAlmostEqual(point["states"]["reactant"], -23.4)
            self.assertAlmostEqual(point["states"]["product"], -22.4)
            self.assertAlmostEqual(point["pairs"][0]["orthogonalized_coupling_ry"],
                                   0.25)
            self.assertEqual(point["adiabatic_energies_ry"], [-23.5, -22.3])
            table = (work / "fde_pes.tsv").read_text(encoding="utf-8")
            self.assertIn("reactant_energy_ry\tproduct_energy_ry", table)
            table_values = table.splitlines()[1].split("\t")
            self.assertAlmostEqual(float(table_values[-3]), 0.8)
            self.assertAlmostEqual(float(table_values[-2]), -22.7)
            self.assertAlmostEqual(float(table_values[-1]), 0.25)
            for state in ("reactant", "product"):
                checkpoint_path = work / "g0" / state / "checkpoint.json"
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                self.assertTrue(checkpoint["converged"])
                self.assertEqual(checkpoint["cycle"], 2)
                self.assertIn(f"/{state}/", checkpoint["densities"]["F"])
                self.assertFalse((checkpoint_path.parent / "cycle-001").exists())
                input_text = (checkpoint_path.parent / "cycle-002" / "F" / "INPUT").read_text(
                    encoding="utf-8")
                self.assertIn("ks_solver                genelpa", input_text)
            self.assertFalse(list(work.glob("**/*-CHARGE-DENSITY.restart")))
            logs = list(work.glob("g0/*/cycle-*/*/fde_abacus.log"))
            logs.append(work / "g0" / "postprocess" / "fde_postprocess.log")
            self.assertTrue(all("OMP_NUM_THREADS=1" in path.read_text(encoding="utf-8")
                                for path in logs))


if __name__ == "__main__":
    unittest.main()
