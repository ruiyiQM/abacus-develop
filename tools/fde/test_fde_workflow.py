import importlib.util
import json
from pathlib import Path
import struct
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

    def test_rks_accepts_only_closed_shell_fragment_assignments(self):
        parameters = fde_workflow.embedded_scf_spin_parameters(
            {"spin_mode": "rks"}, 4, 0, 0)
        self.assertEqual(parameters, {"nspin": 1, "nelec": 4, "nupdown": 0})

        spec = self.spec()
        spec["controls"] = {"spin_mode": "rks"}
        with self.assertRaisesRegex(
                fde_workflow.WorkflowError, "odd-electron fragments"):
            fde_workflow.validate_spec(spec)

    def test_uks_remains_the_default_spin_mode(self):
        self.assertEqual(
            fde_workflow.embedded_scf_spin_parameters({}, 7, 0, 1),
            {"nspin": 2, "nelec": 7, "nupdown": 1})
        spec = self.spec()
        spec["controls"] = {"spin_mode": "invalid"}
        with self.assertRaisesRegex(fde_workflow.WorkflowError, "spin_mode"):
            fde_workflow.validate_spec(spec)

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

    def test_merges_extended_stage_and_recovery_mixing_controls(self):
        controls = {
            "mixing_type": "broyden",
            "mixing_ndim": 12,
            "fragment_mixing": {"F": {"mixing_beta": 0.2}},
            "mixing_recovery": {
                "enabled": True,
                "fallbacks": [{"mixing_type": "plain", "mixing_beta": 0.05}],
            },
        }
        stage = {
            "name": "medium",
            "mixing": {"mixing_gg0": 1.0},
            "fragment_mixing": {"F": {"mixing_beta_mag": 0.1}},
        }
        self.assertEqual(
            fde_workflow.fragment_mixing_parameters(controls, "F", stage, 1),
            {"mixing_type": "plain", "mixing_beta": 0.05,
             "mixing_beta_mag": 0.1, "mixing_ndim": 12,
             "mixing_gg0": 1.0})

    def test_selects_adaptive_scf_stage_from_previous_outer_residual(self):
        controls = {
            "maximum_freeze_thaw_cycles": 12,
            "strict_confirmation_cycles": 2,
            "adaptive_scf": {
                "enabled": True,
                "force_strict_cycle": 10,
                "stages": [
                    {"name": "loose", "minimum_density_rms": 1e-3,
                     "maximum_iterations": 20, "density_tolerance": 1e-4,
                     "strict": False},
                    {"name": "medium", "minimum_density_rms": 1e-5,
                     "maximum_iterations": 60, "density_tolerance": 1e-6,
                     "strict": False},
                    {"name": "strict", "minimum_density_rms": 0.0,
                     "maximum_iterations": 200, "density_tolerance": 1e-8,
                     "strict": True},
                ],
            },
        }
        self.assertEqual(fde_workflow.scf_schedule(controls, 1)["mode"], "loose")
        self.assertEqual(
            fde_workflow.scf_schedule(controls, 2, 2e-4)["mode"], "medium")
        self.assertEqual(
            fde_workflow.scf_schedule(controls, 3, 2e-7)["mode"], "strict")
        self.assertEqual(
            fde_workflow.scf_schedule(controls, 10, 1.0)["mode"], "strict")

    def test_validates_adaptive_scf_and_recovery_policy(self):
        spec = self.spec()
        spec["controls"] = {
            "allow_partial_scf": True,
            "maximum_freeze_thaw_cycles": 8,
            "strict_confirmation_cycles": 2,
            "adaptive_scf": {
                "enabled": True,
                "force_strict_cycle": 7,
                "stages": [
                    {"name": "loose", "minimum_density_rms": 1e-3,
                     "maximum_iterations": 20, "density_tolerance": 1e-4,
                     "strict": False, "mixing": {"mixing_ndim": 8}},
                    {"name": "strict", "minimum_density_rms": 0.0,
                     "maximum_iterations": 100, "density_tolerance": 1e-8,
                     "strict": True},
                ],
            },
            "mixing_recovery": {
                "enabled": True,
                "fallbacks": [{"mixing_type": "plain", "mixing_beta": 0.05}],
            },
        }
        fde_workflow.validate_spec(spec)
        spec["controls"]["adaptive_scf"]["stages"][1]["minimum_density_rms"] = 1e-4
        with self.assertRaisesRegex(fde_workflow.WorkflowError, "must be zero"):
            fde_workflow.validate_spec(spec)

    def test_validates_jacobi_and_outer_anderson_controls(self):
        spec = self.spec()
        spec["controls"] = {
            "update_scheme": "jacobi",
            "jacobi_parallelism": 2,
            "update_order": ["CH3Cl", "F"],
            "outer_mixing": {
                "type": "anderson", "beta": 0.4, "history": 4,
                "regularization": 1e-10, "apply_in_strict": False,
            },
        }
        fde_workflow.validate_spec(spec)
        spec["controls"]["outer_mixing"]["apply_in_strict"] = True
        with self.assertRaisesRegex(fde_workflow.WorkflowError, "final artifacts"):
            fde_workflow.validate_spec(spec)

    def test_linear_outer_mixing_preserves_populations_and_writes_binary(self):
        old = {
            "fragment": "F", "state": "reactant", "geometry": "g0",
            "schema_version": 2, "cycle": 1, "scf_converged": True,
            "scf_iterations": 4, "scf_density_residual": 1e-6,
            "grid_size": 2, "grid_dimensions": [2, 1, 1], "cell_volume": 2.0,
            "grid_fingerprint": "grid", "pseudopotentials": "pp",
            "orbitals": "nao", "core_density": "none",
            "functionals": ["pbe", "lc94"], "populations": [1, 0],
            "energies_ry": [0.1, 0.2],
            "alpha": [0.8, 0.2], "beta": [0.0, 0.0],
        }
        raw = dict(old)
        raw["alpha"] = [0.2, 0.8]
        alpha, beta, detail = fde_workflow.mix_density_history(
            [(old, raw)], "linear", 0.5, 1e-10)
        self.assertEqual(list(alpha), [0.5, 0.5])
        self.assertEqual(list(beta), [0.0, 0.0])
        self.assertEqual(detail["coefficients"], [1.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.fde_density"
            fde_workflow.write_mixed_density(path, raw, alpha, beta)
            mixed = fde_workflow.read_density(path)
            self.assertEqual(mixed["artifact_format"], "binary")
            self.assertFalse(mixed["scf_converged"])
            self.assertEqual(mixed["populations"], [1, 0])
            self.assertEqual(list(mixed["alpha"]), [0.5, 0.5])

    def test_anderson_coefficients_are_normalized(self):
        base = {
            "fragment": "F", "state": "reactant", "geometry": "g0",
            "grid_size": 2, "cell_volume": 2.0, "populations": [1, 0],
            "alpha": [0.7, 0.3], "beta": [0.0, 0.0],
        }
        first_raw = dict(base)
        first_raw["alpha"] = [0.6, 0.4]
        second_old = dict(base)
        second_old["alpha"] = [0.6, 0.4]
        second_raw = dict(base)
        second_raw["alpha"] = [0.55, 0.45]
        alpha, _, detail = fde_workflow.mix_density_history(
            [(base, first_raw), (second_old, second_raw)],
            "anderson", 0.5, 1e-8)
        self.assertAlmostEqual(sum(detail["coefficients"]), 1.0)
        self.assertAlmostEqual(sum(alpha), 1.0)
        self.assertTrue(all(value >= 0.0 for value in alpha))

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

    def test_reads_versioned_binary_density(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary.fde_density"
            prefix = (
                b"FDE_DENSITY_BINARY 1 2\n"
                b"FRAGMENT F\nSTATE reactant\nGEOMETRY g0\n"
                b"GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS pp\nORBITALS nao\n"
                b"CORE_DENSITY none\nFUNCTIONALS pbe lc94\n"
                b"GRID 2 1 1 2\nPOPULATIONS 1 1\nSCF 3 0 50 0.04\n"
                b"ENERGIES_RY 0 0\nBYTE_ORDER LITTLE_ENDIAN\n"
                b"RHO_ALPHA_BINARY 2\n")
            middle = b"\nRHO_BETA_BINARY 2\n"
            path.write_bytes(prefix + struct.pack("<2d", 0.75, 0.25)
                             + middle + struct.pack("<2d", 0.5, 0.5)
                             + b"\nEND\n")

            density = fde_workflow.read_density(path)
            self.assertEqual(density["schema_version"], 2)
            self.assertEqual(density["cycle"], 3)
            self.assertFalse(density["scf_converged"])
            self.assertEqual(list(density["alpha"]), [0.75, 0.25])
            self.assertEqual(list(density["beta"]), [0.5, 0.5])

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

    def test_reads_abacus_json_scf_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            output = job / "OUT.test"
            output.mkdir()
            (output / "abacus.json").write_text(json.dumps({
                "output": [{"scf": [
                    {"energy": -10.0, "drho": 0.2, "time": 1.5},
                    {"energy": -11.0, "drho": 0.01, "time": 2.0},
                ]}]
            }), encoding="utf-8")
            metrics = fde_workflow.read_abacus_scf_metrics(job)
            self.assertTrue(metrics["abacus_json_available"])
            self.assertEqual(metrics["electronic_steps"], 2)
            self.assertAlmostEqual(metrics["electronic_step_time_seconds"], 3.5)
            self.assertAlmostEqual(metrics["initial_drho"], 0.2)
            self.assertAlmostEqual(metrics["final_drho"], 0.01)


if __name__ == "__main__":
    unittest.main()
