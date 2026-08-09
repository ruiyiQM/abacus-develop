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

    def test_canonical_energy_counts_shared_terms_once(self):
        artifacts = [
            {"subsystem_total_energy_ry": -10.0, "ion_ion_energy_ry": 2.0,
             "hartree_cross_energy_ry": 0.4, "nonadditive_kinetic_energy_ry": 0.2,
             "nonadditive_xc_energy_ry": -0.1},
            {"subsystem_total_energy_ry": -12.0, "ion_ion_energy_ry": 2.0,
             "hartree_cross_energy_ry": 0.5, "nonadditive_kinetic_energy_ry": 0.3,
             "nonadditive_xc_energy_ry": -0.2},
        ]
        self.assertAlmostEqual(fde_workflow.canonical_two_fragment_energy(artifacts, 1e-12),
                               -23.4)


if __name__ == "__main__":
    unittest.main()
