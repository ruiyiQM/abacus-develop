#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


configure_module = load_module("configure_slurm_workflow", "configure_slurm_workflow.py")
collect_module = load_module("collect_fde_results", "collect_fde_results.py")


class ScalingScriptContractTest(unittest.TestCase):
    def test_scaling_runner_requires_explicit_template_and_current_binary_name(self):
        runner = (ROOT / "run_fde_mpi_scaling.sbatch").read_text(
            encoding="utf-8")
        submitter = (ROOT / "submit_fde_mpi_scaling.sh").read_text(
            encoding="utf-8")
        combined = runner + submitter

        self.assertIn("prepared scaling template directory is required", runner)
        self.assertIn("ABACUS_FDE_SCALING_TEMPLATE", submitter)
        self.assertIn("install/bin/abacus", combined)
        self.assertNotIn("abacus_std_para", combined)
        self.assertNotIn("test_abacus/fodft_parallel_validation", combined)


class ConfigureSlurmWorkflowTest(unittest.TestCase):
    def test_configure_adds_explicit_slurm_commands_and_provenance(self):
        source = {
            "schema_version": 1,
            "controls": {"spin_mode": "uks", "ks_solver": "genelpa"},
            "provenance": {"case": "unit-test"},
        }
        binary = Path("/tmp/abacus-test")
        result = configure_module.configure(
            source,
            binary=binary,
            ranks=20,
            threads=4,
            postprocess_ranks=1,
            srun="/cluster/apps/slurm/bin/srun",
            exclusive=True,
            persistent_session=True,
            work_directory=Path("/cluster/scratch/test/work"),
            device="gpu",
            ks_solver="cusolver",
            binary_sha256="abc123",
        )

        self.assertEqual(source["controls"]["ks_solver"], "genelpa")
        self.assertEqual(
            result["abacus_command"],
            [
                "/cluster/apps/slurm/bin/srun",
                "--exclusive",
                "--cpu-bind=cores",
                "--distribution=block:block",
                "--ntasks=20",
                "--cpus-per-task=4",
                "/tmp/abacus-test",
            ],
        )
        self.assertIn("--ntasks=1", result["postprocess_command"])
        self.assertIn("--overlap", result["session_command"])
        self.assertIn("--exact", result["session_command"])
        self.assertNotIn("--exclusive", result["session_command"])
        self.assertEqual(
            result["controls"]["execution_mode"], "persistent_session"
        )
        self.assertEqual(result["controls"]["device"], "gpu")
        self.assertEqual(result["controls"]["ks_solver"], "cusolver")
        self.assertEqual(
            result["provenance"]["euler_slurm"]["ranks_per_subsystem"], 20
        )
        self.assertEqual(
            result["provenance"]["euler_slurm"]["binary_sha256"], "abc123"
        )

    def test_atomic_writer_requires_explicit_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "workflow.json"
            output.write_text("old\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                configure_module.write_json_atomic(output, {"new": True}, False)
            configure_module.write_json_atomic(output, {"new": True}, True)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"new": True})


class CollectFdeResultsTest(unittest.TestCase):
    @staticmethod
    def write_tsv(path: Path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(rows)

    def make_point(self, root: Path, task_id: int, geometry: str):
        work = root / "runs" / geometry
        state = work / geometry / "reactant"
        state.mkdir(parents=True)
        (state / "checkpoint.json").write_text(
            json.dumps(
                {
                    "converged": True,
                    "cycle": 5,
                    "density_rms": 1.0e-6,
                    "energy_change_ry": 2.0e-6,
                    "energy_ry": -10.0 - task_id,
                }
            ),
            encoding="utf-8",
        )
        (state / "performance.json").write_text(
            json.dumps(
                {
                    "total_scf_iterations": 42 + task_id,
                    "total_wall_time_seconds": 12.5 + task_id,
                }
            ),
            encoding="utf-8",
        )
        self.write_tsv(
            work / "fde_pes.tsv",
            [
                ["geometry", "coordinate_angstrom", "reactant_energy_ry"],
                [geometry, str(task_id / 10), str(-10.0 - task_id)],
            ],
        )
        spec = root / "specs" / f"{geometry}.json"
        spec.parent.mkdir(exist_ok=True)
        spec.write_text(
            json.dumps(
                {
                    "work_directory": str(work),
                    "geometries": [{"label": geometry}],
                    "states": [{"label": "reactant"}],
                }
            ),
            encoding="utf-8",
        )
        return spec

    def test_collects_pes_convergence_and_specs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_point(root, 0, "g00")
            second = self.make_point(root, 1, "g01")
            manifest = root / "spec_manifest.tsv"
            self.write_tsv(
                manifest,
                [
                    ["task_id", "geometry", "spec"],
                    ["0", "g00", str(first)],
                    ["1", "g01", str(second)],
                ],
            )
            output = root / "final"

            summary = collect_module.collect(manifest, output)

            self.assertEqual(summary["complete_tasks"], 2)
            self.assertEqual(summary["pes_rows"], 2)
            _, pes_rows = collect_module.read_tsv(output / "fde_pes.tsv")
            self.assertEqual([row[0] for row in pes_rows], ["g00", "g01"])
            convergence_text = (output / "convergence.tsv").read_text(encoding="utf-8")
            self.assertIn("42", convergence_text)
            self.assertIn("43", convergence_text)
            self.assertEqual(len(list((output / "specs").glob("*.json"))), 2)
            collection = json.loads(
                (output / "collection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(collection["fde_pes_sha256"]), 64)

    def test_incomplete_task_fails_without_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = self.make_point(root, 0, "g00")
            missing_work = root / "runs" / "g01"
            missing_work.mkdir(parents=True)
            incomplete = root / "specs" / "g01.json"
            incomplete.write_text(
                json.dumps(
                    {
                        "work_directory": str(missing_work),
                        "geometries": [{"label": "g01"}],
                        "states": [{"label": "reactant"}],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "spec_manifest.tsv"
            self.write_tsv(
                manifest,
                [
                    ["task_id", "spec_path"],
                    ["0", str(complete)],
                    ["1", str(incomplete)],
                ],
            )
            with self.assertRaises(RuntimeError):
                collect_module.collect(manifest, root / "strict")

            summary = collect_module.collect(
                manifest, root / "partial", allow_incomplete=True
            )
            self.assertEqual(summary["complete_tasks"], 1)
            self.assertEqual(summary["incomplete_tasks"], 1)

    def test_refuses_to_replace_existing_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_point(root, 0, "g00")
            manifest = root / "spec_manifest.tsv"
            self.write_tsv(manifest, [["task_id", "spec"], ["0", str(spec)]])
            output = root / "final"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                collect_module.collect(manifest, output)


if __name__ == "__main__":
    unittest.main()
