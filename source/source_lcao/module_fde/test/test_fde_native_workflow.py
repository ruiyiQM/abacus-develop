"""Bounded real two-state freeze--thaw/coupling developer checks, not benchmarks."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_bounded(command, timeout):
    with subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Give MPI launchers a chance to terminate their ranks before forcing exit.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def prepare_case(root, repository, executable, case, launcher):
    rks = case == "molecule_rks"
    periodic = case.startswith("periodic")
    cell = 8 if periodic else 24
    grid = 24 if periodic else 48
    resources = repository / "tests/PP_ORB"
    pp = resources / "H_ONCV_PBE-1.0.upf"
    orbital = resources / "H_gga_8au_100Ry_1s.orb"
    template = root / "template"
    template.mkdir()
    separation = 4 if periodic else 6
    positions = [(cell / 2 + side * separation / 2, cell / 2 + offset, cell / 2)
                 for side in (-1, 1) for offset in (-0.7, 0.7)]
    coordinates = "\n".join(" ".join(str(x / cell) for x in point) + " 0 0 0"
                              for point in positions)
    structure = (f"ATOMIC_SPECIES\nH 1 {pp.name}\nNUMERICAL_ORBITAL\n{orbital.name}\n"
                 f"LATTICE_CONSTANT\n1\nLATTICE_VECTORS\n{cell} 0 0\n0 {cell} 0\n0 0 {cell}\n"
                 f"ATOMIC_POSITIONS\nDirect\nH\n0\n{len(positions)}\n{coordinates}\n")
    (template / "STRU").write_text(structure, encoding="utf-8")
    (template / "KPT").write_text("K_POINTS\n0\nGamma\n1 1 1 0 0 0\n")
    (template / "INPUT").write_text(
        f"INPUT_PARAMETERS\necutwfc 10\nnx {grid}\nny {grid}\nnz {grid}\n"
        "nbands 2\nsmearing_method fixed\n"
        f"pseudo_dir {resources}\norbital_dir {resources}\n")
    neutral = 2
    states = []
    seeds = {}
    for name, charged in (("left", "A"), ("right", "B")):
        assignments = {label: {"charge": 0, "spin": 0} for label in ("A", "B")}
        assignments[charged] = {"charge": -2 if rks else -1, "spin": 0 if rks else 1}
        state = {"label": name, "total_charge": -2 if rks else -1,
                 "total_spin": 0 if rks else 1, "fragments": assignments}
        states.append(state)
        seeds[name] = {}
        for label, assignment in assignments.items():
            electrons = neutral - assignment["charge"]
            alpha = (electrons + assignment["spin"]) // 2
            beta = electrons - alpha
            volume = cell ** 3
            path = root / f"{name}_{label}.fde_seed"
            path.write_text(
                f"FDE_UNIFORM_DENSITY_SEED 1\nFRAGMENT {label}\nSTATE {name}\n"
                f"GEOMETRY {sha256(template / 'STRU')}\nGRID_FINGERPRINT grid-{grid}\n"
                f"PSEUDOPOTENTIALS {sha256(pp)}\nORBITALS {sha256(orbital)}\n"
                f"CORE_DENSITY none\nFUNCTIONALS pbe pw91k\nGRID {grid} {grid} {grid} {volume}\n"
                f"POPULATIONS {alpha} {beta}\nRHO_UNIFORM {alpha / volume:.17g} {beta / volume:.17g}\nEND\n")
            seeds[name][label] = str(path)
    spec = {
        "schema_version": 1, "abacus_command": launcher or [str(executable)],
        "postprocess_command": [str(executable)], "run_postprocess": True,
        "work_directory": str(root / "work"),
        "fragments": [{"label": label, "neutral_valence_electrons": neutral,
                       "atom_indices": list(range(index * neutral, (index + 1) * neutral))}
                      for index, label in enumerate(("A", "B"))],
        "states": states,
        "controls": {"spin_mode": "rks" if rks else "uks",
                     "ks_solver": "scalapack_gvx" if launcher else "lapack", "kpar": 1,
                     "maximum_freeze_thaw_cycles": 4, "maximum_scf_iterations": 40,
                     "scf_density_tolerance": 1e-8, "strict_confirmation_cycles": 2,
                     "freeze_thaw_density_tolerance": 1e-7, "energy_tolerance_ry": 1e-8,
                     "mixing_type": "plain", "mixing_beta": 1.0, "mixing_beta_mag": 1.0,
                     "mixing_gg0": 0.0, "mixing_gg0_mag": 0.0,
                     "retain_completed_cycles": 1, "remove_abacus_restart_files": True},
        "geometries": [{"label": "pair", "coordinate_angstrom": 0.0,
                        "template_directory": str(template), "initial_densities": seeds}],
    }
    return spec, {str(path.name): sha256(path) for path in (pp, orbital)}


def compare_mpi_reports(reports):
    """Compare fixed-basis observables; determinant signs may differ across solves."""
    assert len(reports) == 3, "expected all 1/2/4-rank reports"
    tolerances = {"energy": 1e-8, "hamiltonian": 1e-8, "coupling": 1e-8,
                  "overlap": 1e-10, "particles": 1e-8, "projector": 1e-6}

    def values(report):
        result = {}
        for key, field in (("hamiltonian", "hamiltonian_ry"), ("overlap", "overlap")):
            for i, value in enumerate(report[field]):
                result[(key, str(i))] = abs(value) if i in (1, 2) else value
        result[("coupling", "HS")] = report["hamiltonian_ry"][1] * report["overlap"][1]
        for i, value in enumerate(report["coupling"]["adiabatic_energies_ry"]):
            result[("energy", "root" + str(i))] = value
        pair = report["coupling"]["pairs"][0]
        result[("coupling", "orthogonalized")] = abs(pair["orthogonalized_coupling_ry"])
        for state in report["states"]:
            label = state["label"]
            result[("energy", label)] = state["energy_ry"]
            for fragment, particles in state["particles"].items():
                for spin, number in enumerate(particles):
                    result[("particles", f"{label}/{fragment}/{spin}")] = number
            for fragment, spins in state["occupied"].items():
                for spin, columns in spins.items():
                    dimension = len(columns[0]) if columns else 0
                    for row in range(dimension):
                        for column in range(dimension):
                            result[("projector", f"{label}/{fragment}/{spin}/{row}/{column}")] = math.fsum(
                                orbital[row] * orbital[column] for orbital in columns)
        return result

    reference = values(reports[0])
    maximum = {key: 0.0 for key in tolerances}
    for ranks, report in zip((1, 2, 4), reports):
        assert report["passed"] and report["native_processes"] == ranks, report
        for key in ("binary_sha256", "resources_sha256", "source_files_sha256", "input_sha256", "controls"):
            assert report[key] == reports[0][key], key
        current = values(report)
        assert current.keys() == reference.keys()
        for key, value in current.items():
            delta = abs(value - reference[key])
            maximum[key[0]] = max(maximum[key[0]], delta)
            assert delta <= tolerances[key[0]], (ranks, key, delta, tolerances[key[0]])
    return {"passed": True, "ranks": [1, 2, 4], "tolerances": tolerances,
            "maximum_deltas": maximum,
            "convention": "S/H off-diagonal magnitudes and occupied C C^T projectors are sign/gauge invariant in the fixed AO basis; postprocessing uses one rank."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--case", choices=("molecule_uks", "molecule_rks", "periodic_gamma_uks"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit", help="Git SHA for a verified source archive without .git")
    parser.add_argument("--launcher", nargs=argparse.REMAINDER, default=[],
                        help="complete fragment command including the ABACUS executable")
    args = parser.parse_args()
    if "{ranks}" in args.launcher:
        reports = []
        with tempfile.TemporaryDirectory(prefix="fde-native-mpi-") as directory:
            for ranks in (1, 2, 4):
                output = Path(directory) / f"rank{ranks}.json"
                launcher = [item.replace("{ranks}", str(ranks)) for item in args.launcher]
                command = [sys.executable, str(Path(__file__).resolve()), str(args.executable),
                           str(args.repository), "--case", args.case, "--output", str(output)]
                if args.source_commit:
                    command += ["--source-commit", args.source_commit]
                command += ["--launcher"] + launcher
                completed = subprocess.run(command, text=True, capture_output=True, timeout=140)
                if not output.is_file():
                    raise RuntimeError(completed.stdout[-2000:] + completed.stderr[-2000:])
                reports.append(json.loads(output.read_text()))
                if completed.returncode != 0:
                    break
            try:
                result = compare_mpi_reports(reports)
            except AssertionError as error:
                result = {"passed": False, "error": str(error)}
            result.update({"case": args.case, "runs": reports, "scientific_acceptance": "not_evaluated"})
            rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output:
                args.output.write_text(rendered, encoding="utf-8")
            print(rendered)
            return 0 if result["passed"] else 1
    repository = args.repository.resolve()
    executable = args.executable.resolve()
    sys.path.insert(0, str(repository / "tools/fde"))
    from fde_workflow import read_density, read_fragment, read_postprocess_result
    source_head = args.source_commit or subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    report = {"schema_version": 1, "case": args.case, "passed": False,
              "scope": "minimal-basis charged finite-cell developer model; not physical acceptance",
              "source_head": source_head,
              "binary_sha256": sha256(executable),
              "version": subprocess.check_output([str(executable), "--version"], text=True, timeout=10).strip(),
              "launcher": args.launcher, "postprocess_ranks": 1, "kpar": 1,
              "omp_threads": os.environ.get("OMP_NUM_THREADS"),
              "test_script_sha256": sha256(Path(__file__)),
              "thresholds": {"particle_error": 1e-8, "trace_error": 1e-8,
                             "reciprocity_error": 1e-8, "maximum_local_bytes": 200 * 1024 ** 2}}
    report["entry_command"] = [sys.executable, str(Path(__file__).resolve()), str(executable),
                               str(repository), "--case", args.case]
    if args.source_commit:
        report["entry_command"] += ["--source-commit", args.source_commit]
    if args.launcher:
        report["entry_command"] += ["--launcher"] + args.launcher
    sources = ("tools/fde/fde_workflow.py",
               "source/source_estate/module_charge/chg_rhog_io.cpp",
               "source/source_lcao/module_fde/embedding/fde_lcao_driver.cpp",
               "source/source_lcao/module_fde/runtime/fde_session.cpp",
               "source/source_lcao/module_fde/coupling/fde_diabatic_postprocess.cpp",
               "source/source_lcao/module_fde/coupling/fde_electronic_coupling.cpp",
               "source/source_lcao/module_fde/coupling/fde_linearized_state.cpp")
    report["source_files_sha256"] = {name: sha256(repository / name) for name in sources}
    with tempfile.TemporaryDirectory(prefix="fde-native-workflow-") as directory:
        root = Path(directory)
        try:
            spec, report["resources_sha256"] = prepare_case(root, repository, executable, args.case, args.launcher)
            report["input_sha256"] = {str(p.relative_to(root)): sha256(p)
                                       for p in sorted(root.rglob("*")) if p.is_file()}
            path = root / "workflow.json"
            path.write_text(json.dumps(spec, indent=2) + "\n")
            report["controls"] = spec["controls"]
            command = [sys.executable, str(repository / "tools/fde/fde_workflow.py"), "run", str(path)]
            report["command"] = command
            run = run_bounded(command, 120)
            report["returncode"] = run.returncode
            assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
            work = root / "work/pair"
            report["states"] = []
            for state in spec["states"]:
                checkpoint = json.loads((work / state["label"] / "checkpoint.json").read_text())
                assert checkpoint["converged"] and checkpoint["strict_confirmations"] >= 2
                assert checkpoint["cycle"] >= 2
                record = {"label": state["label"], "energy_ry": checkpoint["energy_ry"],
                          "cycles": checkpoint["cycle"], "particles": {}, "occupied": {}}
                for label, density_path in checkpoint["densities"].items():
                    density = read_density(Path(density_path))
                    integral = [math.fsum(density[spin]) * density["cell_volume"] / density["grid_size"]
                                for spin in ("alpha", "beta")]
                    for actual, expected in zip(integral, density["populations"]):
                        assert abs(actual - expected) <= 1e-8, (label, integral)
                    record["particles"][label] = integral
                    fragment = read_fragment(Path(checkpoint["fragments"][label]))
                    if args.case == "molecule_rks":
                        assert density["alpha"] == density["beta"]
                        assert fragment["alpha"] == fragment["beta"]
                    record["occupied"][label] = {spin: [orb["coefficients"] for orb in fragment[spin]]
                                                  for spin in ("alpha", "beta")}
                report["states"].append(record)
            result_path = work / "postprocess/fde_diabatic.fde_diabatic"
            result = read_postprocess_result(result_path, [state["label"] for state in spec["states"]])
            report["coupling"] = result
            assert len(result["pairs"]) == 1 and len(result["adiabatic_energies_ry"]) == 2
            pair = result["pairs"][0]
            assert pair["overlap_reciprocity_error"] <= 1e-8, pair
            assert pair["maximum_transition_density_trace_error"] <= 1e-8, pair
            records = {line.split()[0]: line.split()[1:] for line in result_path.read_text().splitlines() if line.split()}
            report["hamiltonian_ry"] = [float(x) for x in records["HAMILTONIAN_RY"][1:]]
            report["overlap"] = [float(x) for x in records["OVERLAP"][1:]]
            assert report["hamiltonian_ry"][1] == report["hamiltonian_ry"][2]
            assert report["overlap"][1] == report["overlap"][2]
            assert float(records["MAXIMUM_RESIDUAL"][0]) <= 1e-8
            report["output_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            assert report["output_bytes"] < report["thresholds"]["maximum_local_bytes"]
            first_log = sorted(root.rglob("fde_abacus.log"))[0]
            lines = first_log.read_text().splitlines()
            heading = next(i for i, line in enumerate(lines) if "THREADS/PROC" in line)
            report["native_processes"] = int(lines[heading + 1].split()[2])
            report["passed"] = True
        except Exception as error:
            report["error"] = str(error)
            logs = sorted(root.rglob("fde_abacus.log")) + sorted(root.rglob("fde_postprocess.log"))
            report["failure_logs"] = {str(p.relative_to(root)): p.read_text(errors="replace")[-2500:]
                                      for p in logs[-3:]}
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
