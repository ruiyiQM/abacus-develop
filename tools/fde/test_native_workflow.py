"""Failure and gauge controls for the native developer MPI comparison."""

import copy
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


PATH = (Path(__file__).resolve().parents[2]
        / "source/source_lcao/module_fde/test/test_fde_native_workflow.py")
SPEC = importlib.util.spec_from_file_location("native_workflow", PATH)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def reports():
    record = {"passed": True, "binary_sha256": "binary", "resources_sha256": {},
              "source_files_sha256": {}, "input_sha256": {}, "controls": {},
              "hamiltonian_ry": [1.0, 0.2, 0.2, 2.0], "overlap": [1.0, 0.1, 0.1, 1.0],
              "coupling": {"adiabatic_energies_ry": [0.9, 2.1],
                           "pairs": [{"orthogonalized_coupling_ry": 0.05}]},
              "states": [{"label": "left", "energy_ry": 1.0, "particles": {"A": [2.0, 0.0]},
                          "occupied": {"A": {"alpha": [[1.0, 0.0], [0.0, 1.0]], "beta": []}}}]}
    result = [copy.deepcopy(record) for _ in range(3)]
    for ranks, item in zip((1, 2, 4), result):
        item["native_processes"] = ranks
    return result


class NativeWorkflowComparisonTest(unittest.TestCase):
    def test_timeout_terminates_the_workflow_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            code = ("import pathlib,subprocess,sys,time; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
            with self.assertRaises(subprocess.TimeoutExpired):
                native.run_bounded([sys.executable, "-c", code, str(pid_file)], 0.5)
            pid = int(pid_file.read_text())
            state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                   text=True, capture_output=True).stdout.strip()
            if state and not state.startswith("Z"):
                os.kill(pid, signal.SIGKILL)
                self.fail("timed-out workflow left a live child: " + state)

    def test_accepts_occupied_rotations_and_consistent_determinant_sign_changes(self):
        data = reports()
        data[1]["states"][0]["occupied"]["A"]["alpha"] = [[0.6, 0.8], [-0.8, 0.6]]
        for name in ("hamiltonian_ry", "overlap"):
            for index in (1, 2):
                data[1][name][index] *= -1
        self.assertTrue(native.compare_mpi_reports(data)["passed"])

    def test_rejects_changed_state_energy_subspace_or_coupling(self):
        for target in ("energy", "subspace", "coupling", "relative_sign"):
            with self.subTest(target=target):
                data = reports()
                if target == "energy":
                    data[2]["states"][0]["energy_ry"] += 1e-5
                elif target == "subspace":
                    data[2]["states"][0]["occupied"]["A"]["alpha"][0][0] += 1e-3
                elif target == "coupling":
                    data[2]["coupling"]["pairs"][0]["orthogonalized_coupling_ry"] += 1e-5
                else:
                    data[2]["hamiltonian_ry"][1] *= -1
                with self.assertRaises(AssertionError):
                    native.compare_mpi_reports(data)

    def test_complex_phase_uses_conjugate_projectors_and_hs_product(self):
        data = reports()
        for record in data:
            record["hamiltonian_ry"] = [1, [0.2, 0.1], [0.2, -0.1], 2]
            record["overlap"] = [1, [0.1, 0.03], [0.1, -0.03], 1]
            record["coupling"]["pairs"][0]["orthogonalized_coupling_ry"] = [0.04, 0.03]
            record["states"][0]["occupied"]["A"]["alpha"] = [[[0.6, 0], [0, 0.8]]]
        data[1]["states"][0]["occupied"]["A"]["alpha"] = [[[0, 0.6], [-0.8, 0]]]
        for name in ("hamiltonian_ry", "overlap"):
            for index in (1, 2):
                value = native.scalar(data[1][name][index]) * (1j if index == 1 else -1j)
                data[1][name][index] = native.json_scalar(value)
        data[1]["coupling"]["pairs"][0]["orthogonalized_coupling_ry"] = [-0.03, 0.04]
        self.assertTrue(native.compare_mpi_reports(data)["passed"])
        data[1]["hamiltonian_ry"][1][0] += 0.01
        with self.assertRaises(AssertionError):
            native.compare_mpi_reports(data)

    def test_rejects_incomplete_execution_or_mixed_input_binary_and_resources(self):
        for target in ("passed", "native_processes", "binary_sha256", "resources_sha256", "input_sha256"):
            with self.subTest(target=target):
                data = reports()
                data[1][target] = False if target == "passed" else "mismatch"
                with self.assertRaises(AssertionError):
                    native.compare_mpi_reports(data)
        with self.assertRaises(AssertionError):
            native.compare_mpi_reports(reports()[:2])

    def test_boundary_limits_require_same_states_and_conjugate_occupied_spaces(self):
        data = [copy.deepcopy(reports()[0]) for _ in range(5)]
        for report in data:
            report["state_definition"] = [{"total_charge": 1, "total_spin": 1}]
            report["coupling"]["pairs"][0]["overlap"] = 0.1
        data[2]["states"][0]["occupied"]["A"]["alpha"] = [[[0.6, 0], [0, 0.8]]]
        data[3]["states"][0]["occupied"]["A"]["alpha"] = [[[0.6, 0], [0, -0.8]]]
        self.assertTrue(native.compare_boundary_reports(data)["passed"])
        data[3]["states"][0]["occupied"]["A"]["alpha"][0][1] = [0, 0.8]
        with self.assertRaises(AssertionError):
            native.compare_boundary_reports(data)


if __name__ == "__main__":
    unittest.main()
