"""Complex artifact protocol controls; native physics checks use the runtime harness."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import fde_workflow as workflow
import test_fde_workflow as real_tests


def fragment_text(state, label):
    index = 0 if label == "F" else 1
    coefficients = "(1,0) (0,0)" if index == 0 else "(0,0) (1,0)"
    return (f"FDE_FRAGMENT_SCF_ARTIFACT 2\nSTATE {state}\nFRAGMENT {label}\n"
            "GEOMETRY geometry\nORBITAL_BASIS basis\nBOUNDARY_TWIST 0.17 0.11 0\n"
            "RESOURCES context\nDENSITY_PATH unused\nCYCLE 2\nSCF_CONVERGED 1\n"
            f"AO_DIMENSION 2\nACTIVE_ORBITALS 1 {index}\n"
            "SUBSYSTEM_TOTAL_ENERGY_RY -1\nION_ION_ENERGY_RY 0\n"
            "HARTREE_CROSS_ENERGY_RY 0\nNONADDITIVE_KINETIC_ENERGY_RY 0\n"
            "NONADDITIVE_XC_ENERGY_RY 0\n"
            f"ALPHA 1\nORBITAL {label} -0.2 {coefficients}\nBETA 0\n"
            "AO_OVERLAP 4 (1,0) (0,0.2) (0,-0.2) (1,0)\n"
            "HAMILTONIAN_ALPHA_RY 4 (2,0) (0.3,0.4) (0.3,-0.4) (1,0)\n"
            "HAMILTONIAN_BETA_RY 4 (2,0) (0.3,0.4) (0.3,-0.4) (1,0)\n"
            "END_FDE_FRAGMENT_SCF_ARTIFACT\n")


# Exact C++ local-state hand model; protocol parsing does not recompute its answers.
RESULT = ("FDE_DIABATIC_RESULT 2\nGEOMETRY geometry\nORBITALS basis\n"
          "BOUNDARY_TWIST 0.17 0.11 0\nRESOURCES context\n"
          "COUPLING_PROVIDER symmetric_linearized\nSTATES 2 reactant product\n"
          "HAMILTONIAN_RY 4 (5,0) (-0.3,3.85) (-0.3,-3.85) (4.4,0)\n"
          "OVERLAP 4 (1,0) (0,0.8) (0,-0.8) (1,0)\n"
          "PAIR 0 1 (0,0.8) (-0.3,3.85) (-0.5,0.25) 0 0 0 0\n"
          "ADIABATIC_ENERGIES_RY 2 3.75 5.25\n"
          "COEFFICIENTS 4 (1,0) (0,1) (1,0) (0,-1)\nMAXIMUM_RESIDUAL 1e-15\nEND\n")


class ComplexWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spec = real_tests.FdeWorkflowTest().spec()
        self.spec["controls"] = {"boundary_twist": [0.17, 0.11, 0]}
        self.results = []
        for state in self.spec["states"]:
            paths = {}
            for label in ("F", "CH3Cl"):
                path = self.root / f"{state['label']}-{label}.fragment"
                path.write_text(fragment_text(state["label"], label))
                paths[label] = str(path)
            files = workflow.compose_state(state["label"], [Path(p) for p in paths.values()], -2, self.root)
            self.results.append(dict(files, state=state["label"], converged=True,
                                     energy_ry=-2, fragments=paths))

    def test_boundary_controls_runtime_and_complex_composition_preserve_imaginary_values(self):
        workflow.validate_spec(self.spec)
        path = self.root / "config"
        workflow.write_runtime_config(path, self.spec, self.spec["states"][0], "F",
                                      {label: self.root / label for label in ("F", "CH3Cl")}, "out", 10, 1e-8)
        self.assertIn("BOUNDARY_TWIST 0.17000000000000001 0.11 0", path.read_text())
        determinant = Path(self.results[0]["determinant"]).read_text()
        linearized = Path(self.results[0]["linearized_state"]).read_text()
        self.assertTrue(determinant.startswith("FDE_DIABATIC_DETERMINANT 2\n"))
        self.assertIn("RESOURCES context\n", determinant)
        self.assertIn("(0.29999999999999999,0.40000000000000002)", linearized)
        config = workflow.write_postprocess_inputs(self.spec, {}, self.results, self.root)
        self.assertIn("BOUNDARY_TWIST", config.read_text())
        overlap = (self.root / "ao_overlap.fde_matrix").read_text()
        self.assertTrue(overlap.startswith("FDE_AO_MATRIX 2\nGEOMETRY geometry\nORBITALS basis\n"))
        self.assertIn("(0,0.20000000000000001)", overlap)

    def test_boundary_validation_rejects_malformed_nonfinite_and_noncanonical(self):
        for twist in ([0.1, 0], [0, 0, 0.5], [True, 0, 0], [0, float("nan"), 0], "0 0 0"):
            with self.subTest(twist=twist):
                self.spec["controls"]["boundary_twist"] = twist
                with self.assertRaisesRegex(workflow.WorkflowError, "boundary_twist"):
                    workflow.validate_spec(self.spec)

    def test_all_fragments_including_first_are_checked_against_requested_state(self):
        paths = [Path(p) for p in self.results[0]["fragments"].values()]
        with self.assertRaisesRegex(workflow.WorkflowError, "another state"):
            workflow.compose_state("product", paths, -2, self.root)

    def test_postprocess_rejects_context_mixing_in_any_state_before_writing(self):
        path = Path(self.results[1]["fragments"]["CH3Cl"])
        original = path.read_text()
        for before, after in (("RESOURCES context", "RESOURCES other"),
                              ("GEOMETRY geometry", "GEOMETRY other"),
                              ("0.17 0.11 0", "0.18 0.11 0"),
                              ("(0,0.2)", "(0,0.3)"),
                              ("STATE product", "STATE reactant")):
            with self.subTest(before=before):
                path.write_text(original.replace(before, after))
                with self.assertRaises(workflow.WorkflowError):
                    workflow.write_postprocess_inputs(self.spec, {}, self.results, self.root)
        path.write_text(original)
        real_spec = copy.deepcopy(self.spec)
        del real_spec["controls"]["boundary_twist"]
        with self.assertRaisesRegex(workflow.WorkflowError, "boundary"):
            workflow.write_postprocess_inputs(real_spec, {}, self.results, self.root)

    def test_fragment_parser_rejects_truncation_nonfinite_and_duplicate_metadata(self):
        path = Path(self.results[0]["fragments"]["F"])
        original = path.read_text()
        for text in (original.replace("END_FDE_FRAGMENT_SCF_ARTIFACT", "END"),
                     original.replace("(0,0.2)", "(0,nan)"),
                     original.replace("AO_DIMENSION 2", "AO_DIMENSION 3"),
                     original.replace("RESOURCES context", "RESOURCES context\nRESOURCES other")):
            path.write_text(text)
            with self.assertRaises(workflow.WorkflowError):
                workflow.read_fragment(path)

    def test_complex_result_is_json_safe_and_bound_to_workflow_resources(self):
        path = self.root / "result"
        path.write_text(RESULT)
        result = workflow.read_postprocess_result(path, ["reactant", "product"])
        self.assertEqual(result["pairs"][0]["orthogonalized_coupling_ry"], [-0.5, 0.25])
        self.assertEqual(result["pairs"][0]["overlap_magnitude"], 0.8)
        self.assertEqual(result["hamiltonian_ry"][1], [-0.3, 3.85])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        workflow._validate_postprocess_context(self.spec, self.results, result)
        for key, value in (("resources", "other"), ("boundary_twist", [0, 0, 0]),
                           ("artifact_version", 1), ("maximum_residual", 1e-4)):
            changed = dict(result, **{key: value})
            with self.assertRaises(workflow.WorkflowError):
                workflow._validate_postprocess_context(self.spec, self.results, changed)

    def test_complex_result_rejects_nonhermitian_inconsistent_or_incomplete_records(self):
        path = self.root / "result"
        for before, after in (("(-0.3,-3.85)", "(-0.3,3.85)"),
                              ("PAIR 0 1 (0,0.8)", "PAIR 0 1 (0,0.7)"),
                              ("(0,0.8)", "(0,nan)"),
                              ("RESOURCES context\n", ""),
                              ("MAXIMUM_RESIDUAL 1e-15", "MAXIMUM_RESIDUAL -1"),
                              ("ADIABATIC_ENERGIES_RY 2 3.75 5.25", "ADIABATIC_ENERGIES_RY 1 3.75")):
            with self.subTest(before=before):
                path.write_text(RESULT.replace(before, after))
                with self.assertRaises(workflow.WorkflowError):
                    workflow.read_postprocess_result(path, ["reactant", "product"])


if __name__ == "__main__":
    unittest.main()
