import importlib.util
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("fde_workflow.py")
SPEC = importlib.util.spec_from_file_location("fde_workflow", MODULE_PATH)
fde_workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fde_workflow)


FAKE_ABACUS = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys


def value(path, key):
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].lower() == key.lower():
            return fields[1]
    raise RuntimeError("missing " + key)


request_config = Path(sys.argv[1]) if len(sys.argv) == 2 else None
cwd = request_config.parent if request_config is not None else Path.cwd()
task = ("embedded_scf" if request_config is not None
        else value(cwd / "INPUT", "fde_task"))
print("OMP_NUM_THREADS=" + os.environ.get("OMP_NUM_THREADS", ""))
if task == "embedded_session":
    print("FDE_SESSION_READY 1", flush=True)
    request_index = 0
    for command in sys.stdin:
        fields = command.split()
        if fields == ["STOP"]:
            print("FDE_SESSION_STOPPED", flush=True)
            raise SystemExit(0)
        if len(fields) != 3 or fields[0] != "RUN":
            print("FDE_SESSION_ERROR protocol invalid request", flush=True)
            raise SystemExit(2)
        completed = subprocess.run(
            [sys.executable, __file__, fields[2]],
            cwd=Path(fields[2]).parent,
            check=False,
        )
        if completed.returncode != 0:
            print("FDE_SESSION_ERROR " + fields[1] + " child failed", flush=True)
            raise SystemExit(2)
        mode = ("density_seed" if request_index == 0
                else "resident_ao_density_matrix_and_orbitals")
        print("FDE_SESSION_DONE " + fields[1] + " " + mode + " "
              + str(request_index), flush=True)
        request_index += 1
    raise SystemExit(0)
if task == "diabatic_postprocess":
    config = Path(value(cwd / "INPUT", "fde_config"))
    text = config.read_text(encoding="utf-8")
    if text.count("DETERMINANT ") != 2 or text.count("LINEARIZED_STATE ") != 2:
        raise RuntimeError("postprocess config is incomplete")
    prefix = cwd / value(config, "OUTPUT_PREFIX")
    Path(str(prefix) + ".fde_diabatic").write_text(
        "FDE_DIABATIC_RESULT 1\n"
        "COUPLING_PROVIDER symmetric_linearized\n"
        "STATES 2 reactant product\n"
        "HAMILTONIAN_RY 4 -23.4 -22.7 -22.7 -22.4\n"
        "OVERLAP 4 1 0.8 0.8 1\n"
        "PAIR 0 1 0.8 -22.7 0.25 1e-16 2e-16 0.04 0.016\n"
        "ADIABATIC_ENERGIES_RY 2 -23.5 -22.3\n"
        "COEFFICIENTS 4 1 0 0 1\nMAXIMUM_RESIDUAL 1e-16\nEND\n",
        encoding="utf-8")
    Path(str(prefix) + ".fde_diabatic.tsv").write_text("kind\tfirst\tsecond\npair\t0\t1\n")
    raise SystemExit(0)

config = (request_config if request_config is not None
          else Path(value(cwd / "INPUT", "fde_config")))
state = value(config, "ACTIVE_STATE")
fragment = value(config, "ACTIVE_FRAGMENT")
fragment_xc = value(config, "FRAGMENT_XC")
cycle = int(cwd.parent.name.split("-")[1])
output = cwd / "OUT.fake"
output.mkdir()
(output / "abacus.json").write_text(json.dumps({
    "output": [{"scf": [
        {"energy": -10.0, "ediff": 0.0, "drho": 0.1, "time": 0.02},
        {"energy": -10.1, "ediff": -0.1, "drho": 0.01, "time": 0.03}
    ]}]
}), encoding="utf-8")
(output / "fake-CHARGE-DENSITY.restart").write_text("restart", encoding="utf-8")
neutral = {}
assignment = None
for line in config.read_text(encoding="utf-8").splitlines():
    fields = line.split()
    if fields and fields[0] == "FRAGMENT":
        neutral[fields[1]] = int(fields[2])
    if fields and fields[0] == "STATE" and fields[1] == state:
        for index in range(int(fields[4])):
            offset = 5 + 3 * index
            if fields[offset] == fragment:
                assignment = (int(fields[offset + 1]), int(fields[offset + 2]))
if assignment is None:
    raise RuntimeError("missing active fragment state assignment")
electrons = neutral[fragment] - assignment[0]
alpha = (electrons + assignment[1]) // 2
beta = (electrons - assignment[1]) // 2
partial_first_attempt = ((cwd / "PARTIAL_ON_FIRST_STRICT_ATTEMPT").exists()
                         and "-retry-" not in cwd.name and cycle > 1)
if ((cwd / "PARTIAL_ON_FIRST_CYCLE").exists() and cycle == 1) or partial_first_attempt:
    density = cwd / "result.partial.fde_density"
    density.write_text(
        "FDE_DENSITY_ARTIFACT 2\n"
        f"FRAGMENT {fragment}\nSTATE {state}\nGEOMETRY g0\n"
        "GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS fake\nORBITALS fake\n"
        f"CORE_DENSITY none\nFUNCTIONALS {fragment_xc} lc94\nGRID 1 1 1 1\n"
        f"POPULATIONS {alpha} {beta}\nSCF {cycle} 0 50 0.04\n"
        "ENERGIES_RY 0 0\n"
        f"RHO_ALPHA 1 {alpha}\nRHO_BETA 1 {beta}\nEND\n",
        encoding="utf-8")
    raise SystemExit(0)
density = cwd / "result.fde_density"
density.write_text(
    "FDE_DENSITY_ARTIFACT 1\n"
    f"FRAGMENT {fragment}\nSTATE {state}\nGEOMETRY g0\n"
    "GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS fake\nORBITALS fake\n"
    f"CORE_DENSITY none\nFUNCTIONALS {fragment_xc} lc94\nGRID 1 1 1 1\n"
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
    "ORBITAL_BASIS fake\nDENSITY_PATH result.fde_density\nSCF_CONVERGED 1\n"
    "AO_DIMENSION 4\n"
    f"ACTIVE_ORBITALS 1 {active}\nSUBSYSTEM_TOTAL_ENERGY_RY {subsystem}\n"
    "ION_ION_ENERGY_RY 2\nHARTREE_CROSS_ENERGY_RY 0.5\n"
    "NONADDITIVE_KINETIC_ENERGY_RY 0.3\nNONADDITIVE_XC_ENERGY_RY -0.2\n"
    f"ALPHA 1\nORBITAL {fragment} -1 {alpha_orbital}\n"
    f"BETA 1\nORBITAL {fragment} -0.5 {beta_orbital}\n"
    f"AO_OVERLAP 16 {identity}\nHAMILTONIAN_ALPHA_RY 16 {hamiltonian}\n"
    f"HAMILTONIAN_BETA_RY 16 {hamiltonian}\nEND\n",
    encoding="utf-8")
'''


def write_seed(path: Path, state: str, fragment: str, alpha: int, beta: int,
               fragment_xc: str) -> None:
    path.write_text(
        "FDE_UNIFORM_DENSITY_SEED 1\n"
        f"FRAGMENT {fragment}\nSTATE {state}\nGEOMETRY g0\n"
        "GRID_FINGERPRINT grid\nPSEUDOPOTENTIALS fake\nORBITALS fake\n"
        f"CORE_DENSITY none\nFUNCTIONALS {fragment_xc} lc94\nGRID 1 1 1 1\n"
        f"POPULATIONS {alpha} {beta}\nRHO_UNIFORM {alpha} {beta}\nEND\n",
        encoding="utf-8")


def prepare_case(root: Path, partial_first_cycle: bool = False,
                 rks: bool = False, adaptive: bool = False,
                 jacobi_outer: bool = False, gpu: bool = False,
                 persistent_session: bool = False,
                 fragment_xc: str = "pbe"):
    fake_abacus = root / "fake_abacus.py"
    fake_abacus.write_text(FAKE_ABACUS, encoding="utf-8")
    template = root / "template"
    template.mkdir()
    (template / "INPUT").write_text("INPUT_PARAMETERS\n", encoding="utf-8")
    if partial_first_cycle:
        (template / "PARTIAL_ON_FIRST_CYCLE").write_text("1\n", encoding="utf-8")
    if adaptive:
        (template / "PARTIAL_ON_FIRST_STRICT_ATTEMPT").write_text(
            "1\n", encoding="utf-8")
    seeds = root / "seeds"
    seeds.mkdir()
    state_definitions = [
        {"label": "reactant", "total_charge": -1, "total_spin": 0,
         "fragments": {"F": {"charge": -1, "spin": 0},
                       "CH3Cl": {"charge": 0, "spin": 0}}},
        {"label": "product", "total_charge": -1, "total_spin": 0,
         "fragments": ({"F": {"charge": 1, "spin": 0},
                        "CH3Cl": {"charge": -2, "spin": 0}}
                       if rks else
                       {"F": {"charge": 0, "spin": 1},
                        "CH3Cl": {"charge": -1, "spin": -1}})},
    ]
    neutral = {"F": 7, "CH3Cl": 14}
    populations = {}
    for state in state_definitions:
        for fragment, assignment in state["fragments"].items():
            populations[(state["label"], fragment)] = fde_workflow.spin_population(
                neutral[fragment], assignment["charge"], assignment["spin"])
    initial = {"reactant": {}, "product": {}}
    for (state, fragment), (alpha, beta) in populations.items():
        seed = seeds / f"{state}_{fragment}.fde_seed"
        write_seed(seed, state, fragment, alpha, beta, fragment_xc)
        initial[state][fragment] = str(seed)

    work = root / "work"
    controls = {"maximum_freeze_thaw_cycles": 4 if partial_first_cycle else 3,
                "freeze_thaw_density_tolerance": 1e-12,
                "energy_tolerance_ry": 1e-12,
                "ks_solver": "genelpa", "kpar": 1,
                "retain_completed_cycles": 3 if partial_first_cycle else 1,
                "remove_abacus_restart_files": True,
                "update_order": ["F", "CH3Cl"]}
    controls.update({"fragment_xc": fragment_xc, "embedding_xc": "pbe"})
    if gpu:
        controls.update({"device": "gpu", "ks_solver": "cusolver"})
    if rks:
        controls["spin_mode"] = "rks"
    if persistent_session:
        controls["execution_mode"] = "persistent_session"
    if partial_first_cycle:
        controls.update({"allow_partial_scf": True,
                         "inexact_freeze_thaw_cycles": 1,
                         "inexact_scf_iterations": 50,
                         "inexact_scf_density_tolerance": 1e-3,
                         "maximum_scf_iterations": 200,
                         "scf_density_tolerance": 1e-8,
                         "strict_confirmation_cycles": 2})
    if adaptive:
        controls.update({
            "allow_partial_scf": True,
            "maximum_freeze_thaw_cycles": 4,
            "strict_confirmation_cycles": 2,
            "retain_completed_cycles": 3,
            "adaptive_scf": {
                "enabled": True,
                "force_strict_cycle": 3,
                "stages": [
                    {"name": "loose", "minimum_density_rms": 1e-3,
                     "maximum_iterations": 10, "density_tolerance": 1e-4,
                     "strict": False, "mixing": {"mixing_type": "broyden"}},
                    {"name": "strict", "minimum_density_rms": 0.0,
                     "maximum_iterations": 20, "density_tolerance": 1e-8,
                     "strict": True, "mixing": {"mixing_type": "pulay"}},
                ],
            },
            "mixing_recovery": {
                "enabled": True,
                "fallbacks": [{"mixing_type": "plain", "mixing_beta": 0.05}],
            },
        })
    if jacobi_outer:
        controls.update({
            "update_scheme": "jacobi",
            "jacobi_parallelism": 2,
            "outer_mixing": {
                "type": "anderson", "beta": 0.5, "history": 2,
                "regularization": 1e-10, "apply_in_strict": False,
            },
        })
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
        "states": state_definitions,
        "controls": controls,
        "geometries": [{"label": "g0", "coordinate_angstrom": 0.2,
                        "template_directory": str(template),
                        "initial_densities": initial}],
    }
    if persistent_session:
        specification["session_command"] = [sys.executable, str(fake_abacus)]
    spec_path = root / "workflow.json"
    spec_path.write_text(json.dumps(specification), encoding="utf-8")
    return work, spec_path


class FdeWorkflowEndToEndTest(unittest.TestCase):
    def test_update_scheme_case_preserves_gauss_seidel_dependencies(self):
        for scheme in ("gauss_seidel", "GAUSS_SEIDEL", "Auto"):
            with self.subTest(scheme=scheme), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                work, spec_path = prepare_case(root)
                spec = json.loads(spec_path.read_text())
                spec["controls"].update({"update_scheme": scheme, "retain_completed_cycles": 3})
                spec_path.write_text(json.dumps(spec))
                fde_workflow.run_workflow(spec_path)
                config = (work / "g0/reactant/cycle-001/CH3Cl/FDE_CONFIG").read_text()
                self.assertIn("cycle-001/F/result.fde_density", config)
                checkpoint = json.loads((work / "g0/reactant/checkpoint.json").read_text())
                self.assertEqual(checkpoint["update_scheme"], "gauss_seidel")

    def test_postprocess_rejects_bad_new_results_and_preserves_published_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root)
            fde_workflow.run_workflow(spec_path)
            published = work / "g0/postprocess/fde_diabatic.fde_diabatic"
            valid = published.read_text()
            cases = {
                "missing": None,
                "empty": "",
                "table_missing": valid,
                "header": valid.replace("FDE_DIABATIC_RESULT 1", "OTHER_RESULT 1"),
                "states": valid.replace("STATES 2 reactant product", "STATES 2 reactant other"),
                "end": valid.removesuffix("END\n"),
                "pair_missing": "\n".join(line for line in valid.splitlines() if not line.startswith("PAIR ")) + "\n",
                "pair_duplicate": valid.replace("END\n", "PAIR 0 1 0.8 -22.7 0.25\nEND\n"),
                "pair_index": valid.replace("PAIR 0 1", "PAIR 0 7"),
                "nan": valid.replace("0.25 1e-16", "nan 1e-16"),
                "roots": valid.replace("ADIABATIC_ENERGIES_RY 2 -23.5 -22.3", "ADIABATIC_ENERGIES_RY 2 -23.5"),
                "matrix": valid.replace("OVERLAP 4", "OVERLAP 3"),
                "matrix_infinite": valid.replace("HAMILTONIAN_RY 4 -23.4", "HAMILTONIAN_RY 4 inf"),
                "coefficients": valid.replace("COEFFICIENTS 4", "COEFFICIENTS 3"),
            }
            for name, data in cases.items():
                with self.subTest(case=name):
                    def postprocess(command, cwd, **_kwargs):
                        config = Path(next(line.split()[1] for line in (cwd / "INPUT").read_text().splitlines()
                                           if line.startswith("fde_config")))
                        prefix = cwd / next(line.split()[1] for line in config.read_text().splitlines()
                                            if line.startswith("OUTPUT_PREFIX"))
                        if data is not None:
                            Path(str(prefix) + ".fde_diabatic").write_text(data)
                        if name != "table_missing":
                            Path(str(prefix) + ".fde_diabatic.tsv").write_text("table\n")
                        return subprocess.CompletedProcess(command, 0)
                    with mock.patch.object(fde_workflow.subprocess, "run", side_effect=postprocess):
                        with self.assertRaises(fde_workflow.WorkflowError):
                            fde_workflow.run_workflow(spec_path)
                    self.assertEqual(published.read_text(), valid)

    def test_postprocess_accepts_legacy_pairs_and_reduced_overlap_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root)
            fde_workflow.run_workflow(spec_path)
            valid = (work / "g0/postprocess/fde_diabatic.fde_diabatic").read_text()
            legacy = valid.replace("COUPLING_PROVIDER symmetric_linearized\n", "").replace(
                "0.25 1e-16 2e-16 0.04 0.016", "0.25")
            reduced = valid.replace("ADIABATIC_ENERGIES_RY 2 -23.5 -22.3",
                                    "ADIABATIC_ENERGIES_RY 1 -23.5").replace(
                                        "COEFFICIENTS 4 1 0 0 1", "COEFFICIENTS 2 1 0")
            for text, roots in ((legacy, 2), (reduced, 1)):
                path = root / "result"
                path.write_text(text)
                result = fde_workflow.read_postprocess_result(path, ["reactant", "product"])
                self.assertEqual(len(result["pairs"]), 1)
                self.assertEqual(len(result["adiabatic_energies_ry"]), roots)

    def test_nonfinite_coordinate_cannot_publish_pes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root)
            spec = json.loads(spec_path.read_text())
            spec["geometries"][0]["coordinate_angstrom"] = float("nan")
            spec_path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(fde_workflow.WorkflowError, "non-finite"):
                fde_workflow.run_workflow(spec_path)
            self.assertFalse((work / "fde_pes.json").exists())

    def test_relative_launch_scripts_invalidate_completed_checkpoints(self):
        for persistent in (False, True):
            with self.subTest(persistent_session=persistent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                work, spec_path = prepare_case(root, persistent_session=persistent)
                spec = json.loads(spec_path.read_text())
                command = [sys.executable, "../../../../../fake_abacus.py"]
                spec["abacus_command"] = command
                if persistent:
                    spec["session_command"] = command
                spec["postprocess_command"] = [sys.executable, "../../../fake_abacus.py"]
                spec_path.write_text(json.dumps(spec))
                fde_workflow.run_workflow(spec_path)
                script = root / "fake_abacus.py"
                script.write_text(script.read_text() + "\n# changed executable identity\n")
                with self.assertRaisesRegex(fde_workflow.WorkflowError, "checkpoint inputs changed"):
                    fde_workflow.run_workflow(spec_path)

    def test_relative_executables_use_their_actual_launch_directories(self):
        for persistent in (False, True):
            with self.subTest(persistent_session=persistent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                work, spec_path = prepare_case(root, persistent_session=persistent)
                runner = root / "template/runner.py"
                runner.write_text(FAKE_ABACUS)
                runner.chmod(0o755)
                (root / "fake_abacus.py").chmod(0o755)
                spec = json.loads(spec_path.read_text())
                spec["abacus_command"] = ["./runner.py"]
                if persistent:
                    spec["session_command"] = ["./runner.py"]
                spec["postprocess_command"] = ["../../../fake_abacus.py"]
                spec_path.write_text(json.dumps(spec))
                fde_workflow.run_workflow(spec_path)
                self.assertTrue((work / "g0/postprocess/fde_diabatic.fde_diabatic").is_file())

    def test_checkpoint_fingerprint_tracks_scf_policy_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            state = spec["states"][0]
            geometry = spec["geometries"][0]
            driver = root / "driver" / "fde_workflow.py"
            policy = driver.parent / "workflow" / "scf_policy.py"
            policy.parent.mkdir(parents=True)
            driver.write_bytes(MODULE_PATH.read_bytes())
            policy.write_text("original SCF policy", encoding="utf-8")

            def fingerprint():
                return fde_workflow.checkpoint_input_fingerprint(
                    spec, geometry, state, geometry["initial_densities"][state["label"]],
                    work / "g0" / state["label"])

            with mock.patch.object(fde_workflow, "__file__", str(driver)):
                original = fingerprint()
                policy.write_text("changed SCF policy", encoding="utf-8")
                self.assertNotEqual(fingerprint(), original)

    def test_fingerprint_resolves_relative_resources_and_launched_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            state = spec["states"][0]
            geometry = spec["geometries"][0]
            state_directory = work / "g0" / state["label"]
            resource = state_directory / "shared.upf"
            resource.parent.mkdir(parents=True)
            resource.write_text("original pseudo", encoding="utf-8")
            (root / "template" / "INPUT").write_text(
                "INPUT_PARAMETERS\npseudo_dir ../..\n", encoding="utf-8")
            (root / "template" / "STRU").write_text(
                "ATOMIC_SPECIES\nH 1 shared.upf\nLATTICE_CONSTANT\n1\n", encoding="utf-8")
            executable = root / "checkpoint-test-abacus"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            spec["abacus_command"] = ["env", executable.name]

            def fingerprint():
                return fde_workflow.checkpoint_input_fingerprint(
                    spec, geometry, state, geometry["initial_densities"][state["label"]],
                    state_directory)

            with mock.patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ["PATH"]}):
                original = fingerprint()
                resource.write_text("changed pseudo", encoding="utf-8")
                self.assertNotEqual(fingerprint(), original)
                resource.write_text("original pseudo", encoding="utf-8")
                self.assertEqual(fingerprint(), original)
                executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
                self.assertNotEqual(fingerprint(), original)

    def test_restart_checks_input_resources_and_executable_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root)
            resources = root / "resources"
            resources.mkdir()
            pseudo = resources / "H.upf"
            orbital = resources / "H.orb"
            pseudo.write_text("pseudopotential\n", encoding="utf-8")
            orbital.write_text("numerical orbital\n", encoding="utf-8")
            template = root / "template"
            (template / "INPUT").write_text(
                f"INPUT_PARAMETERS\npseudo_dir {resources}\norbital_dir {resources}\n",
                encoding="utf-8")
            (template / "STRU").write_text(
                "ATOMIC_SPECIES\nH 1 H.upf\nNUMERICAL_ORBITAL\nH.orb\n"
                "LATTICE_CONSTANT\n1\nATOMIC_POSITIONS\nDirect\n", encoding="utf-8")
            fde_workflow.run_workflow(spec_path)
            checkpoint = work / "g0" / "reactant" / "checkpoint.json"
            original_checkpoint = checkpoint.read_bytes()
            with mock.patch.object(fde_workflow, "run_fragment_scf", side_effect=AssertionError("SCF rerun")):
                fde_workflow.run_workflow(spec_path)
            self.assertEqual(checkpoint.read_bytes(), original_checkpoint)

            for path in (template / "INPUT", template / "STRU", pseudo, orbital,
                         root / "seeds" / "reactant_F.fde_seed", root / "fake_abacus.py"):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    try:
                        path.write_bytes(original + b"\n# changed in place\n")
                        with self.assertRaisesRegex(fde_workflow.WorkflowError, "checkpoint inputs changed"):
                            fde_workflow.run_workflow(spec_path)
                        self.assertEqual(checkpoint.read_bytes(), original_checkpoint)
                    finally:
                        path.write_bytes(original)
            specification = json.loads(spec_path.read_text(encoding="utf-8"))
            specification["controls"]["scf_density_tolerance"] = 1e-9
            spec_path.write_text(json.dumps(specification), encoding="utf-8")
            with self.assertRaisesRegex(fde_workflow.WorkflowError, "checkpoint inputs changed"):
                fde_workflow.run_workflow(spec_path)

    def test_restart_rejects_legacy_and_damaged_completed_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            work, spec_path = prepare_case(Path(directory))
            fde_workflow.run_workflow(spec_path)
            checkpoint_path = work / "g0" / "reactant" / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            legacy = dict(checkpoint)
            del legacy["input_fingerprint"]
            checkpoint_path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(fde_workflow.WorkflowError, "provenance is absent"):
                fde_workflow.run_workflow(spec_path)
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            for key in ("determinant", "linearized_state"):
                with self.subTest(artifact=key):
                    path = Path(checkpoint[key])
                    original = path.read_bytes()
                    path.write_bytes(original + b"corrupted\n")
                    with self.assertRaisesRegex(fde_workflow.WorkflowError, "checkpoint artifacts changed"):
                        fde_workflow.run_workflow(spec_path)
                    path.unlink()
                    with self.assertRaisesRegex(fde_workflow.WorkflowError, "artifact is missing"):
                        fde_workflow.run_workflow(spec_path)
                    path.write_bytes(original)

    def test_interrupted_finalization_keeps_previous_checkpoint_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            work, spec_path = prepare_case(Path(directory))
            atomic_text = fde_workflow._atomic_text

            def interrupt_linearized(path, value):
                if path.suffix == ".fde_linearized_state":
                    raise OSError("simulated final artifact write failure")
                atomic_text(path, value)

            with mock.patch.object(fde_workflow, "_atomic_text", side_effect=interrupt_linearized):
                with self.assertRaisesRegex(OSError, "simulated final artifact"):
                    fde_workflow.run_workflow(spec_path)
            state_directory = work / "g0" / "reactant"
            checkpoint_path = state_directory / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertFalse(checkpoint["converged"])
            self.assertEqual(checkpoint["cycle"], 1)
            self.assertTrue((state_directory / "cycle-001").is_dir())
            specification = json.loads(spec_path.read_text(encoding="utf-8"))
            original_spec = spec_path.read_bytes()
            specification["geometries"][0]["coordinate_angstrom"] = 0.3
            spec_path.write_text(json.dumps(specification), encoding="utf-8")
            with self.assertRaisesRegex(fde_workflow.WorkflowError, "checkpoint inputs changed"):
                fde_workflow.run_workflow(spec_path)
            spec_path.write_bytes(original_spec)
            fde_workflow.run_workflow(spec_path)
            completed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertTrue(completed["converged"])
            self.assertEqual(completed["cycle"], 2)
            self.assertTrue(Path(completed["linearized_state"]).is_file())
            self.assertFalse((state_directory / "cycle-001").exists())

    def test_persistent_session_reuses_fixed_fragment_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root, persistent_session=True)

            fde_workflow.run_workflow(spec_path)

            for state in ("reactant", "product"):
                state_directory = work / "g0" / state
                workers = sorted((state_directory / "session-workers").glob("*"))
                self.assertEqual(len(workers), 2)
                for worker in workers:
                    log = (worker / "fde_session.log").read_text(encoding="utf-8")
                    self.assertEqual(log.count("FDE_SESSION_READY"), 1)
                    self.assertEqual(log.count("FDE_SESSION_DONE"), 2)
                performance = json.loads(
                    (state_directory / "cycle-002" / "F"
                     / "fde_performance.json").read_text(encoding="utf-8"))
                self.assertTrue(performance["session_reused"])
                self.assertEqual(performance["execution_mode"],
                                 "persistent_session")
                self.assertEqual(
                    performance["warm_start_mode"],
                    "resident_ao_density_matrix_and_orbitals")
                self.assertEqual(performance["abacus_ionic_step"], 1)

    def test_gpu_controls_reach_every_fragment_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root, gpu=True)

            fde_workflow.run_workflow(spec_path)

            inputs = list(work.glob("g0/*/cycle-*/*/INPUT"))
            self.assertTrue(inputs)
            for input_path in inputs:
                text = input_path.read_text(encoding="utf-8")
                self.assertRegex(text, r"(?m)^device\s+gpu$")
                self.assertRegex(text, r"(?m)^ks_solver\s+cusolver$")

    def test_pbe0_in_pbe_reaches_inputs_configs_and_densities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root, fragment_xc="pbe0")

            fde_workflow.run_workflow(spec_path)

            inputs = list(work.glob("g0/*/cycle-*/*/INPUT"))
            configs = list(work.glob("g0/*/cycle-*/*/FDE_CONFIG"))
            densities = list(work.glob("g0/*/cycle-*/*/result.fde_density"))
            self.assertTrue(inputs and configs and densities)
            for input_path in inputs:
                self.assertRegex(input_path.read_text(encoding="utf-8"),
                                 r"(?m)^dft_functional\s+pbe0$")
            for config_path in configs:
                text = config_path.read_text(encoding="utf-8")
                self.assertIn("FRAGMENT_XC pbe0\n", text)
                self.assertIn("EMBEDDING_XC pbe\n", text)
            for density_path in densities:
                self.assertIn("FUNCTIONALS pbe0 lc94\n",
                              density_path.read_text(encoding="utf-8"))

    def test_closed_shell_rks_generates_single_spin_channel_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root, rks=True)

            fde_workflow.run_workflow(spec_path)

            inputs = list(work.glob("g0/*/cycle-*/*/INPUT"))
            self.assertTrue(inputs)
            for input_path in inputs:
                text = input_path.read_text(encoding="utf-8")
                self.assertRegex(text, r"(?m)^nspin\s+1$")
                self.assertRegex(text, r"(?m)^nupdown\s+0$")
            pes = json.loads((work / "fde_pes.json").read_text(encoding="utf-8"))
            self.assertEqual(len(pes["points"][0]["pairs"]), 1)
            performance = json.loads(
                (work / "fde_performance.json").read_text(encoding="utf-8"))
            self.assertEqual(performance["total_subsystem_calls"], 8)
            self.assertEqual(performance["total_scf_iterations"], 16)
            self.assertEqual(performance["schema_version"], 2)
            self.assertIn("phase_totals_seconds", performance)
            self.assertIn("electronic_step_fraction_of_abacus", performance)
            self.assertAlmostEqual(
                performance["total_electronic_step_time_seconds"], 0.4)

    def test_two_states_freeze_thaw_postprocess_and_pes_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root)

            fde_workflow.run_workflow(spec_path)

            pes = json.loads((work / "fde_pes.json").read_text(encoding="utf-8"))
            point = pes["points"][0]
            self.assertAlmostEqual(point["states"]["reactant"], -23.4)
            self.assertAlmostEqual(point["states"]["product"], -22.4)
            self.assertAlmostEqual(point["pairs"][0]["orthogonalized_coupling_ry"],
                                   0.25)
            self.assertEqual(point["coupling_provider"],
                             "symmetric_linearized")
            self.assertAlmostEqual(
                point["pairs"][0]["estimated_coupling_uncertainty_ry"],
                0.016)
            self.assertEqual(point["adiabatic_energies_ry"], [-23.5, -22.3])
            table = (work / "fde_pes.tsv").read_text(encoding="utf-8")
            self.assertIn("reactant_energy_ry\tproduct_energy_ry", table)
            table_row = next(csv.DictReader(table.splitlines(), delimiter="\t"))
            self.assertAlmostEqual(float(table_row["overlap"]), 0.8)
            self.assertAlmostEqual(float(table_row["h12_ry"]), -22.7)
            self.assertAlmostEqual(
                float(table_row["orthogonalized_coupling_ry"]), 0.25)
            self.assertEqual(table_row["coupling_provider"],
                             "symmetric_linearized")
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
            expected_threads = os.environ.get("OMP_NUM_THREADS", "1")
            self.assertTrue(all(f"OMP_NUM_THREADS={expected_threads}"
                                in path.read_text(encoding="utf-8")
                                for path in logs))

    def test_partial_density_is_passed_without_partial_postprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root, partial_first_cycle=True)
            fde_workflow.run_workflow(spec_path)

            for state in ("reactant", "product"):
                checkpoint = json.loads(
                    (work / "g0" / state / "checkpoint.json").read_text(
                        encoding="utf-8"))
                self.assertTrue(checkpoint["converged"])
                self.assertEqual(checkpoint["cycle"], 3)
                first = checkpoint["history"][0]
                self.assertFalse(first["all_inner_scf_converged"])
                self.assertIsNone(first["energy_ry"])
                self.assertEqual(first["scf_mode"], "inexact")
                self.assertEqual(first["inner_scf"]["F"]["iterations"], 50)
                cycle_two_config = (work / "g0" / state / "cycle-002" / "F"
                                    / "FDE_CONFIG").read_text(encoding="utf-8")
                self.assertIn("partial.fde_density", cycle_two_config)
                cycle_one_input = (work / "g0" / state / "cycle-001" / "F"
                                   / "INPUT").read_text(encoding="utf-8")
                cycle_two_input = (work / "g0" / state / "cycle-002" / "F"
                                   / "INPUT").read_text(encoding="utf-8")
                self.assertRegex(cycle_one_input, r"(?m)^scf_nmax\s+50$")
                self.assertRegex(cycle_one_input, r"(?m)^scf_thr\s+0\.001$")
                self.assertRegex(cycle_two_input, r"(?m)^scf_nmax\s+200$")
                self.assertRegex(cycle_two_input, r"(?m)^scf_thr\s+1e-08$")
                self.assertEqual(checkpoint["strict_confirmations"], 2)
                self.assertFalse(list((work / "g0" / state / "cycle-001").glob(
                    "*/result.fde_fragment")))
            self.assertTrue((work / "g0" / "postprocess"
                             / "fde_diabatic.fde_diabatic").is_file())

    def test_adaptive_schedule_retries_strict_partial_scf_with_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(root, adaptive=True)
            fde_workflow.run_workflow(spec_path)

            for state in ("reactant", "product"):
                state_directory = work / "g0" / state
                checkpoint = json.loads(
                    (state_directory / "checkpoint.json").read_text(encoding="utf-8"))
                self.assertTrue(checkpoint["converged"])
                self.assertEqual([item["scf_mode"] for item in checkpoint["history"]],
                                 ["loose", "strict", "strict"])
                self.assertEqual(
                    checkpoint["history"][1]["inner_scf"]["F"]["retry_count"], 1)
                retry_input = (state_directory / "cycle-002" / "F-retry-01"
                               / "INPUT").read_text(encoding="utf-8")
                self.assertRegex(retry_input, r"(?m)^mixing_type\s+plain$")
                self.assertRegex(retry_input, r"(?m)^mixing_beta\s+0\.05$")

    def test_strict_schedule_recovers_without_accepting_partial_results(self):
        for persistent in (False, True):
            with self.subTest(persistent_session=persistent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                work, spec_path = prepare_case(root, persistent_session=persistent)
                (root / "template" / "PARTIAL_ON_FIRST_STRICT_ATTEMPT").write_text("1\n")
                spec = json.loads(spec_path.read_text())
                spec["controls"].update({
                    "allow_partial_scf": False,
                    "retain_completed_cycles": 3,
                    "mixing_recovery": {
                        "enabled": True,
                        "fallbacks": [{"mixing_type": "plain", "mixing_beta": 0.05}],
                    },
                })
                spec_path.write_text(json.dumps(spec))
                fde_workflow.run_workflow(spec_path)

                for state in ("reactant", "product"):
                    state_directory = work / "g0" / state
                    checkpoint = json.loads((state_directory / "checkpoint.json").read_text())
                    self.assertTrue(checkpoint["converged"])
                    self.assertTrue(checkpoint["all_inner_scf_converged"])
                    for fragment in ("F", "CH3Cl"):
                        attempts = checkpoint["history"][1]["inner_scf"][fragment]["attempts"]
                        self.assertEqual([a["converged"] for a in attempts], [False, True])
                        self.assertEqual(attempts[1]["mixing"]["mixing_type"], "plain")
                        retry = state_directory / "cycle-002" / f"{fragment}-retry-01"
                        self.assertIn("result.partial.fde_density", (retry / "FDE_CONFIG").read_text())
                        self.assertEqual(Path(checkpoint["densities"][fragment]).name,
                                         "result.fde_density")
                        self.assertTrue((retry / "result.fde_fragment").is_file())
                self.assertTrue((work / "g0/postprocess/fde_diabatic.fde_diabatic").is_file())

    def test_strict_schedule_rejects_partial_after_recovery_exhaustion(self):
        for enabled in (False, True):
            with self.subTest(recovery_enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                work, spec_path = prepare_case(root)
                (root / "template" / "PARTIAL_ON_FIRST_CYCLE").write_text("1\n")
                spec = json.loads(spec_path.read_text())
                spec["controls"].update({
                    "allow_partial_scf": False,
                    "mixing_recovery": {
                        "enabled": enabled,
                        "fallbacks": [{"mixing_type": "plain", "mixing_beta": 0.05}],
                    },
                })
                spec_path.write_text(json.dumps(spec))
                with self.assertRaisesRegex(fde_workflow.WorkflowError, "only a partial FDE density"):
                    fde_workflow.run_workflow(spec_path)
                retry = work / "g0/reactant/cycle-001/F-retry-01/result.partial.fde_density"
                self.assertEqual(retry.is_file(), enabled)
                self.assertFalse(list(work.glob("g0/*/checkpoint.json")))
                self.assertFalse(list(work.glob("g0/*/cycle-*/*/result.fde_fragment")))
                self.assertFalse((work / "g0/postprocess").exists())

    def test_parallel_jacobi_uses_cycle_snapshot_and_outer_mixing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, spec_path = prepare_case(
                root, adaptive=True, jacobi_outer=True)
            fde_workflow.run_workflow(spec_path)

            checkpoint = json.loads(
                (work / "g0" / "reactant" / "checkpoint.json").read_text(
                    encoding="utf-8"))
            first = checkpoint["history"][0]
            self.assertEqual(first["update_scheme"], "jacobi")
            self.assertTrue(first["outer_mixing"]["applied"])
            self.assertFalse(checkpoint["history"][1]["outer_mixing"]["applied"])
            mixed_f = Path(first["outer_mixing"]["mixed_densities"]["F"])
            self.assertTrue(mixed_f.is_file())
            cycle_one_ch3cl = (work / "g0" / "reactant" / "cycle-001"
                               / "CH3Cl" / "FDE_CONFIG").read_text(encoding="utf-8")
            self.assertIn("reactant_F.fde_seed", cycle_one_ch3cl)
            cycle_two_f = (work / "g0" / "reactant" / "cycle-002"
                           / "F" / "FDE_CONFIG").read_text(encoding="utf-8")
            self.assertIn("outer-mixed", cycle_two_f)


if __name__ == "__main__":
    unittest.main()
