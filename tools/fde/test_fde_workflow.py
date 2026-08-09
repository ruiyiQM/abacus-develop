import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("fde_workflow.py")
SPEC = importlib.util.spec_from_file_location("fde_workflow", MODULE_PATH)
fde_workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fde_workflow)


class FdeWorkflowTest(unittest.TestCase):
    def spec(self):
        return {
            "schema_version": 1,
            "abacus_command": ["abacus"],
            "fragments": [
                {"label": "F", "neutral_valence_electrons": 7, "atom_indices": [0]},
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
        }

    def test_validates_target_charge_localized_states(self):
        fde_workflow.validate_spec(self.spec())
        self.assertEqual(fde_workflow.spin_population(7, -1, 0), (4, 4))
        self.assertEqual(fde_workflow.spin_population(14, -1, -1), (7, 8))

    def test_rejects_cross_state_or_incomplete_assignments(self):
        spec = self.spec()
        del spec["states"][1]["fragments"]["F"]
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)

    def test_rejects_labels_that_can_escape_work_directory(self):
        spec = self.spec()
        spec["fragments"][0]["label"] = "../F"
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)

    def test_validates_distributed_solver_and_cleanup_controls(self):
        spec = self.spec()
        spec["controls"] = {
            "ks_solver": "genelpa",
            "kpar": 1,
            "retain_completed_cycles": 1,
            "remove_abacus_restart_files": True,
        }
        fde_workflow.validate_spec(spec)
        spec["controls"]["ks_solver"] = "cg"
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)

    def test_validates_partial_scf_control(self):
        spec = self.spec()
        spec["controls"] = {"allow_partial_scf": True}
        fde_workflow.validate_spec(spec)
        spec["controls"]["allow_partial_scf"] = "yes"
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)

    def test_validates_and_selects_fragment_mixing_controls(self):
        spec = self.spec()
        spec["controls"] = {
            "fragment_mixing": {
                "F": {"mixing_type": "broyden", "mixing_beta": 0.1},
                "CH3Cl": {"mixing_type": "plain", "mixing_beta": 0.1,
                          "mixing_beta_mag": 0.05},
            }
        }
        fde_workflow.validate_spec(spec)
        self.assertEqual(
            fde_workflow.fragment_mixing_parameters(spec["controls"], "F"),
            {"mixing_type": "broyden", "mixing_beta": 0.1})
        self.assertEqual(
            fde_workflow.fragment_mixing_parameters(spec["controls"], "CH3Cl"),
            {"mixing_type": "plain", "mixing_beta": 0.1,
             "mixing_beta_mag": 0.05})

        spec["controls"]["fragment_mixing"]["unknown"] = {
            "mixing_type": "plain"
        }
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)

    def test_rejects_invalid_fragment_mixing_values(self):
        spec = self.spec()
        spec["controls"] = {
            "fragment_mixing": {"CH3Cl": {"mixing_type": "diis"}}
        }
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)
        spec["controls"]["fragment_mixing"]["CH3Cl"] = {
            "mixing_type": "plain", "mixing_beta_mag": 0.0
        }
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)

    def test_builds_inexact_then_strict_scf_schedule(self):
        controls = {
            "inexact_freeze_thaw_cycles": 3,
            "inexact_scf_iterations": 50,
            "inexact_scf_density_tolerance": 1e-3,
            "maximum_scf_iterations": 200,
            "scf_density_tolerance": 1e-8,
        }
        early = fde_workflow.scf_schedule(controls, 3)
        final = fde_workflow.scf_schedule(controls, 4)
        self.assertEqual(early["mode"], "inexact")
        self.assertEqual(early["maximum_iterations"], 50)
        self.assertEqual(early["density_tolerance"], 1e-3)
        self.assertEqual(final["mode"], "strict")
        self.assertEqual(final["maximum_iterations"], 200)
        self.assertEqual(final["density_tolerance"], 1e-8)

    def test_rejects_schedule_without_room_for_strict_confirmation(self):
        spec = self.spec()
        spec["controls"] = {
            "allow_partial_scf": True,
            "inexact_freeze_thaw_cycles": 3,
            "maximum_freeze_thaw_cycles": 4,
            "strict_confirmation_cycles": 2,
        }
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.validate_spec(spec)

    def test_canonical_energy_counts_shared_terms_once(self):
        artifacts = [
            {"scf_converged": True, "subsystem_total_energy_ry": -10.0,
             "ion_ion_energy_ry": 2.0,
             "hartree_cross_energy_ry": 0.4, "nonadditive_kinetic_energy_ry": 0.2,
             "nonadditive_xc_energy_ry": -0.1},
            {"scf_converged": True, "subsystem_total_energy_ry": -12.0,
             "ion_ion_energy_ry": 2.0,
             "hartree_cross_energy_ry": 0.5, "nonadditive_kinetic_energy_ry": 0.3,
             "nonadditive_xc_energy_ry": -0.2},
        ]
        self.assertAlmostEqual(fde_workflow.canonical_two_fragment_energy(artifacts, 1e-12),
                               -23.4)

        artifacts[1]["scf_converged"] = False
        with self.assertRaises(fde_workflow.WorkflowError):
            fde_workflow.canonical_two_fragment_energy(artifacts, 1e-12)

    def test_reads_compact_uniform_initial_density(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.fde"
            path.write_text(
                "FDE_UNIFORM_DENSITY_SEED 1\n"
                "FRAGMENT F\nSTATE reactant\nGEOMETRY g0\n"
                "GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS pp\nORBITALS nao\n"
                "CORE_DENSITY none\nFUNCTIONALS pbe lc94\n"
                "GRID 2 1 1 4\nPOPULATIONS 4 4\nRHO_UNIFORM 1 1\nEND\n",
                encoding="utf-8")
            density = fde_workflow.read_density(path)
            self.assertEqual(density["cycle"], 0)
            self.assertEqual(density["alpha"], [1.0, 1.0])
            self.assertEqual(density["beta"], [1.0, 1.0])

    def test_selects_only_opted_in_schema_two_partial_density(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            partial = job / "result.partial.fde_density"
            partial.write_text(
                "FDE_DENSITY_ARTIFACT 2\n"
                "FRAGMENT F\nSTATE reactant\nGEOMETRY g0\n"
                "GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS pp\nORBITALS nao\n"
                "CORE_DENSITY none\nFUNCTIONALS pbe lc94\n"
                "GRID 1 1 1 1\nPOPULATIONS 1 0\nSCF 1 0 50 0.04\n"
                "ENERGIES_RY 0 0\nRHO_ALPHA 1 1\nRHO_BETA 1 0\nEND\n",
                encoding="utf-8")
            with self.assertRaises(fde_workflow.WorkflowError):
                fde_workflow.select_scf_density(job, False)
            path, density = fde_workflow.select_scf_density(job, True)
            self.assertEqual(path, partial)
            self.assertFalse(density["scf_converged"])
            self.assertEqual(density["scf_iterations"], 50)


if __name__ == "__main__":
    unittest.main()
