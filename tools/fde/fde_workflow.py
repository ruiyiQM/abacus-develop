#!/usr/bin/env python3
"""Restartable two-fragment FDE freeze--thaw and PES workflow for ABACUS."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


class WorkflowError(RuntimeError):
    pass


def _token(value: object, description: str) -> str:
    text = str(value)
    if not text or any(character.isspace() for character in text):
        raise WorkflowError(f"{description} must be a nonempty token")
    return text


def _absolute_token(path: Path, description: str) -> str:
    return _token(path.resolve(), description)


def spin_population(neutral_electrons: int, charge: int, spin: int) -> Tuple[int, int]:
    electrons = neutral_electrons - charge
    if electrons < 0 or abs(spin) > electrons or (electrons + spin) % 2:
        raise WorkflowError("fragment charge/spin assignment has an invalid electron parity")
    return (electrons + spin) // 2, (electrons - spin) // 2


def validate_spec(spec: Mapping[str, object]) -> None:
    if int(spec.get("schema_version", 0)) != 1:
        raise WorkflowError("workflow schema_version must be 1")
    command = spec.get("abacus_command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise WorkflowError("abacus_command must be a nonempty JSON string array")
    fragments = spec.get("fragments")
    states = spec.get("states")
    if not isinstance(fragments, list) or len(fragments) != 2:
        raise WorkflowError("the production workflow currently requires exactly two fragments")
    if not isinstance(states, list) or len(states) < 2:
        raise WorkflowError("the workflow requires at least two diabatic states")
    labels = [_token(fragment["label"], "fragment label") for fragment in fragments]
    if len(set(labels)) != len(labels):
        raise WorkflowError("fragment labels must be unique")
    atoms: List[int] = []
    for fragment in fragments:
        indices = fragment.get("atom_indices")
        if not isinstance(indices, list) or not indices:
            raise WorkflowError("each fragment requires atom_indices")
        atoms.extend(int(index) for index in indices)
        if int(fragment.get("neutral_valence_electrons", -1)) < 0:
            raise WorkflowError("neutral_valence_electrons must be nonnegative")
    if sorted(atoms) != list(range(len(atoms))):
        raise WorkflowError("fragment atom_indices must partition zero-based STRU atoms")
    state_labels: List[str] = []
    neutral = {str(fragment["label"]): int(fragment["neutral_valence_electrons"])
               for fragment in fragments}
    for state in states:
        state_labels.append(_token(state["label"], "state label"))
        assignments = state.get("fragments")
        if not isinstance(assignments, dict) or set(assignments) != set(labels):
            raise WorkflowError("every state must assign charge and spin to both fragments")
        charge_sum = 0
        spin_sum = 0
        for label in labels:
            assignment = assignments[label]
            charge = int(assignment["charge"])
            spin = int(assignment["spin"])
            spin_population(neutral[label], charge, spin)
            charge_sum += charge
            spin_sum += spin
        if charge_sum != int(state["total_charge"]) or spin_sum != int(state["total_spin"]):
            raise WorkflowError("state totals do not match fragment charge/spin assignments")
    if len(set(state_labels)) != len(state_labels):
        raise WorkflowError("state labels must be unique")


def _records(path: Path) -> Dict[str, List[List[str]]]:
    records: Dict[str, List[List[str]]] = {}
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            fields = raw_line.split()
            if fields:
                records.setdefault(fields[0], []).append(fields[1:])
    return records


def read_density(path: Path) -> Dict[str, object]:
    records = _records(path)
    try:
        alpha = [float(value) for value in records["RHO_ALPHA"][0][1:]]
        beta = [float(value) for value in records["RHO_BETA"][0][1:]]
        if int(records["RHO_ALPHA"][0][0]) != len(alpha) or int(records["RHO_BETA"][0][0]) != len(beta):
            raise WorkflowError("density vector length is inconsistent")
        grid = records["GRID"][0]
        return {
            "fragment": records["FRAGMENT"][0][0],
            "state": records["STATE"][0][0],
            "geometry": records["GEOMETRY"][0][0],
            "cycle": int(records["SCF"][0][0]),
            "grid_size": len(alpha),
            "cell_volume": float(grid[3]),
            "alpha": alpha,
            "beta": beta,
        }
    except (KeyError, IndexError, ValueError) as error:
        raise WorkflowError(f"malformed density artifact {path}: {error}") from error


def density_rms(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    if (first["fragment"] != second["fragment"]
            or first["state"] != second["state"]
            or first["geometry"] != second["geometry"]
            or first["grid_size"] != second["grid_size"]):
        raise WorkflowError("density residual requires compatible artifacts")
    values = list(first["alpha"]) + list(first["beta"])
    reference = list(second["alpha"]) + list(second["beta"])
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(values, reference))
                     / len(values))


def read_fragment(path: Path) -> Dict[str, object]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.split()]
    if not lines or lines[0] != ["FDE_FRAGMENT_SCF_ARTIFACT", "1"]:
        raise WorkflowError(f"unsupported fragment artifact {path}")
    result: Dict[str, object] = {"alpha": [], "beta": []}
    spin = None
    scalar_names = {
        "SUBSYSTEM_TOTAL_ENERGY_RY", "ION_ION_ENERGY_RY",
        "HARTREE_CROSS_ENERGY_RY", "NONADDITIVE_KINETIC_ENERGY_RY",
        "NONADDITIVE_XC_ENERGY_RY",
    }
    for fields in lines[1:]:
        key = fields[0]
        if key in ("ALPHA", "BETA"):
            spin = key.lower()
            result[f"{spin}_count"] = int(fields[1])
        elif key == "ORBITAL":
            if spin is None:
                raise WorkflowError("ORBITAL record precedes its spin header")
            result[spin].append({"fragment": fields[1],
                                 "energy": float(fields[2]),
                                 "coefficients": [float(value) for value in fields[3:]]})
        elif key in ("STATE", "FRAGMENT", "GEOMETRY", "ORBITAL_BASIS", "DENSITY_PATH"):
            result[key.lower()] = fields[1]
        elif key in ("CYCLE", "AO_DIMENSION"):
            result[key.lower()] = int(fields[1])
        elif key == "ACTIVE_ORBITALS":
            result["active_orbitals"] = [int(value) for value in fields[2:]]
        elif key in ("AO_OVERLAP", "HAMILTONIAN_ALPHA_RY", "HAMILTONIAN_BETA_RY"):
            values = [float(value) for value in fields[2:]]
            if int(fields[1]) != len(values):
                raise WorkflowError(f"{key} length is inconsistent")
            result[key.lower()] = values
        elif key in scalar_names:
            result[key.lower()] = float(fields[1])
    dimension = int(result.get("ao_dimension", 0))
    if dimension <= 0 or any(len(orbital["coefficients"]) != dimension
                             for name in ("alpha", "beta") for orbital in result[name]):
        raise WorkflowError(f"invalid occupied-orbital dimensions in {path}")
    return result


def canonical_two_fragment_energy(artifacts: Sequence[Mapping[str, object]],
                                  tolerance: float) -> float:
    if len(artifacts) != 2:
        raise WorkflowError("canonical runtime energy currently requires two fragment artifacts")
    ion_ion = [float(item["ion_ion_energy_ry"]) for item in artifacts]
    if abs(ion_ion[0] - ion_ion[1]) > tolerance:
        raise WorkflowError("fragment jobs disagree on the ion-ion energy")
    correction_source = artifacts[-1]
    correction = (float(correction_source["hartree_cross_energy_ry"])
                  + float(correction_source["nonadditive_kinetic_energy_ry"])
                  + float(correction_source["nonadditive_xc_energy_ry"]))
    return (sum(float(item["subsystem_total_energy_ry"]) for item in artifacts)
            - ion_ion[0] + correction)


def patch_input(path: Path, values: Mapping[str, object]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    remaining = {key.lower(): str(value) for key, value in values.items()}
    output: List[str] = []
    for line in lines:
        stripped = line.strip()
        fields = stripped.split()
        key = fields[0].lower() if fields and not fields[0].startswith("#") else ""
        if key in remaining:
            leading = line[:len(line) - len(line.lstrip())]
            output.append(f"{leading}{key:<24} {remaining.pop(key)}")
        else:
            output.append(line)
    for key in sorted(remaining):
        output.append(f"{key:<24} {remaining[key]}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def write_runtime_config(path: Path,
                         spec: Mapping[str, object],
                         state: Mapping[str, object],
                         active_label: str,
                         densities: Mapping[str, Path],
                         output_prefix: str) -> None:
    fragments = list(spec["fragments"])
    lines = ["FDE_CONFIG 1", f"ATOM_COUNT {sum(len(f['atom_indices']) for f in fragments)}"]
    for fragment in fragments:
        atoms = " ".join(str(index) for index in fragment["atom_indices"])
        lines.append(f"FRAGMENT {fragment['label']} {fragment['neutral_valence_electrons']} "
                     f"{len(fragment['atom_indices'])} {atoms}")
    for definition in spec["states"]:
        assignments = []
        for fragment in fragments:
            label = fragment["label"]
            value = definition["fragments"][label]
            assignments.extend((label, str(value["charge"]), str(value["spin"])))
        lines.append(f"STATE {definition['label']} {definition['total_charge']} "
                     f"{definition['total_spin']} {len(fragments)} {' '.join(assignments)}")
    lines.extend((f"ACTIVE_STATE {state['label']}",
                  f"ACTIVE_FRAGMENT {active_label}",
                  f"ACTIVE_DENSITY {_absolute_token(densities[active_label], 'active density path')}"))
    for fragment in fragments:
        label = fragment["label"]
        if label != active_label:
            lines.append(f"FROZEN_DENSITY {label} "
                         f"{_absolute_token(densities[label], 'frozen density path')}")
    controls = dict(spec.get("controls", {}))
    update_order = controls.get("update_order", [fragment["label"] for fragment in fragments])
    lines.extend((f"OUTPUT_PREFIX {output_prefix}",
                  f"KEDF {controls.get('kedf', 'lc94')}",
                  f"DENSITY_FLOOR_BOHR3 {controls.get('density_floor_bohr3', 1e-12)}",
                  f"MAX_SCF_ITERATIONS {controls.get('maximum_scf_iterations', 100)}",
                  f"SCF_DENSITY_TOLERANCE {controls.get('scf_density_tolerance', 1e-8)}",
                  f"ELECTRON_TOLERANCE {controls.get('electron_tolerance', 1e-8)}",
                  f"MIXING_BETA {controls.get('mixing_beta', 0.3)}",
                  f"MAX_FREEZE_THAW_CYCLES {controls.get('maximum_freeze_thaw_cycles', 20)}",
                  f"FREEZE_THAW_DENSITY_TOLERANCE {controls.get('freeze_thaw_density_tolerance', 1e-7)}",
                  f"ENERGY_TOLERANCE_RY {controls.get('energy_tolerance_ry', 1e-8)}",
                  f"UPDATE_ORDER {len(update_order)} {' '.join(update_order)}",
                  f"K_STATES {len(spec['states'])} " + " ".join(item["label"] for item in spec["states"]),
                  f"L_FRAGMENTS {len(fragments)} " + " ".join(item["label"] for item in fragments),
                  f"M_FRAGMENTS {len(fragments)} " + " ".join(item["label"] for item in fragments),
                  f"SINGULAR_VALUE_TOLERANCE {controls.get('singular_value_tolerance', 1e-10)}",
                  f"OVERLAP_EIGENVALUE_CUTOFF {controls.get('overlap_eigenvalue_cutoff', 1e-9)}",
                  f"SYMMETRY_TOLERANCE {controls.get('symmetry_tolerance', 1e-10)}",
                  f"RESIDUAL_TOLERANCE {controls.get('residual_tolerance', 1e-8)}",
                  f"MINIMUM_ROOT_OVERLAP {controls.get('minimum_root_overlap', 0.5)}",
                  "CALCULATE_FORCE false",
                  f"HARTREE_RECIPROCITY_TOLERANCE_RY {controls.get('hartree_reciprocity_tolerance_ry', 1e-8)}",
                  "END_FDE_CONFIG"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compose_state(state_label: str,
                  fragment_paths: Sequence[Path],
                  energy_ry: float,
                  output_directory: Path) -> Dict[str, str]:
    fragments = [read_fragment(path) for path in fragment_paths]
    reference = fragments[0]
    dimension = int(reference["ao_dimension"])
    for fragment in fragments[1:]:
        if (fragment["state"] != state_label
                or fragment["geometry"] != reference["geometry"]
                or fragment["orbital_basis"] != reference["orbital_basis"]
                or fragment["ao_dimension"] != dimension
                or fragment["ao_overlap"] != reference["ao_overlap"]):
            raise WorkflowError("final fragment SCF artifacts are incompatible")
    determinant_path = output_directory / f"{state_label}.fde_determinant"
    determinant_lines = ["FDE_DIABATIC_DETERMINANT 1",
                         f"STATE {state_label}",
                         f"GEOMETRY {reference['geometry']}",
                         f"ORBITALS {reference['orbital_basis']}",
                         f"AO_DIMENSION {dimension}"]
    for spin in ("alpha", "beta"):
        orbitals = [orbital for fragment in fragments for orbital in fragment[spin]]
        determinant_lines.append(f"{spin.upper()} {len(orbitals)}")
        for orbital in orbitals:
            determinant_lines.append("ORBITAL " + str(orbital["fragment"]) + " "
                                     + format(float(orbital["energy"]), ".17g") + " "
                                     + " ".join(format(value, ".17g")
                                                for value in orbital["coefficients"]))
    determinant_lines.append("END")
    determinant_path.write_text("\n".join(determinant_lines) + "\n", encoding="utf-8")

    matrix_size = dimension * dimension
    average_alpha = [sum(float(fragment["hamiltonian_alpha_ry"][index]) for fragment in fragments)
                     / len(fragments) for index in range(matrix_size)]
    average_beta = [sum(float(fragment["hamiltonian_beta_ry"][index]) for fragment in fragments)
                    / len(fragments) for index in range(matrix_size)]
    linear_path = output_directory / f"{state_label}.fde_linearized_state"
    linear_lines = ["FDE_LINEARIZED_STATE 1", f"STATE {state_label}",
                    f"GEOMETRY {reference['geometry']}", f"AO_DIMENSION {dimension}",
                    f"REFERENCE_ENERGY_RY {energy_ry:.17g}",
                    "HAMILTONIAN_ALPHA_RY " + str(matrix_size) + " "
                    + " ".join(format(value, ".17g") for value in average_alpha),
                    "HAMILTONIAN_BETA_RY " + str(matrix_size) + " "
                    + " ".join(format(value, ".17g") for value in average_beta), "END"]
    linear_path.write_text("\n".join(linear_lines) + "\n", encoding="utf-8")
    return {"determinant": str(determinant_path.resolve()),
            "linearized_state": str(linear_path.resolve())}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_state(spec: Mapping[str, object],
              geometry: Mapping[str, object],
              state: Mapping[str, object],
              output_directory: Path) -> Dict[str, object]:
    fragments = list(spec["fragments"])
    labels = [fragment["label"] for fragment in fragments]
    controls = dict(spec.get("controls", {}))
    update_order = list(controls.get("update_order", labels))
    maximum_cycles = int(controls.get("maximum_freeze_thaw_cycles", 20))
    density_tolerance = float(controls.get("freeze_thaw_density_tolerance", 1e-7))
    energy_tolerance = float(controls.get("energy_tolerance_ry", 1e-8))
    symmetry_tolerance = float(controls.get("symmetry_tolerance", 1e-10))
    state_directory = output_directory / str(state["label"])
    state_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_directory / "checkpoint.json"
    initial = geometry.get("initial_densities", {}).get(state["label"], state.get("initial_densities", {}))
    if set(initial) != set(labels):
        raise WorkflowError(f"state {state['label']} requires one initial density per fragment")
    densities = {label: Path(initial[label]).resolve() for label in labels}
    start_cycle = 1
    previous_energy = None
    history: List[Dict[str, object]] = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("converged"):
            return checkpoint
        densities = {label: Path(checkpoint["densities"][label]) for label in labels}
        start_cycle = int(checkpoint["cycle"]) + 1
        previous_energy = checkpoint.get("energy_ry")
        history = list(checkpoint.get("history", []))

    neutral = {fragment["label"]: int(fragment["neutral_valence_electrons"])
               for fragment in fragments}
    last_fragments: Dict[str, Path] = {}
    for cycle in range(start_cycle, maximum_cycles + 1):
        old_density_data = {label: read_density(path) for label, path in densities.items()}
        cycle_fragments: List[Mapping[str, object]] = []
        for active_label in update_order:
            assignment = state["fragments"][active_label]
            alpha, beta = spin_population(neutral[active_label],
                                          int(assignment["charge"]),
                                          int(assignment["spin"]))
            job_directory = state_directory / f"cycle-{cycle:03d}" / active_label
            if job_directory.exists():
                shutil.rmtree(job_directory)
            shutil.copytree(Path(geometry["template_directory"]), job_directory)
            config_path = job_directory / "FDE_CONFIG"
            write_runtime_config(config_path, spec, state, active_label, densities, "result")
            patch_input(job_directory / "INPUT", {
                "calculation": "scf", "basis_type": "lcao", "gamma_only": 1,
                "nspin": 2, "noncolin": 0, "lspinorb": 0, "symmetry": 0,
                "dft_functional": "pbe", "ks_solver": "lapack", "kpar": 1,
                "nelec": alpha + beta, "nupdown": alpha - beta,
                "fde_task": "embedded_scf", "fde_config": "FDE_CONFIG",
            })
            log_path = job_directory / "fde_abacus.log"
            environment = dict(os.environ)
            environment.setdefault("OMP_NUM_THREADS", "1")
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(list(spec["abacus_command"]), cwd=job_directory,
                                           env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                raise WorkflowError(f"ABACUS failed in {job_directory}; see {log_path}")
            density_path = job_directory / "result.fde_density"
            fragment_path = job_directory / "result.fde_fragment"
            if not density_path.is_file() or not fragment_path.is_file():
                raise WorkflowError(f"ABACUS did not produce FDE artifacts in {job_directory}")
            densities[active_label] = density_path.resolve()
            last_fragments[active_label] = fragment_path.resolve()
            cycle_fragments.append(read_fragment(fragment_path))
        residual = max(density_rms(old_density_data[label], read_density(densities[label]))
                       for label in labels)
        energy = canonical_two_fragment_energy(cycle_fragments, symmetry_tolerance)
        energy_change = None if previous_energy is None else abs(energy - float(previous_energy))
        converged = residual <= density_tolerance and energy_change is not None and energy_change <= energy_tolerance
        history.append({"cycle": cycle, "density_rms": residual, "energy_ry": energy,
                        "energy_change_ry": energy_change})
        checkpoint: Dict[str, object] = {
            "schema_version": 1, "geometry": geometry["label"], "state": state["label"],
            "cycle": cycle, "converged": converged, "density_rms": residual,
            "energy_ry": energy, "energy_change_ry": energy_change,
            "densities": {label: str(densities[label]) for label in labels},
            "fragments": {label: str(last_fragments[label]) for label in labels},
            "history": history,
        }
        _atomic_json(checkpoint_path, checkpoint)
        if converged:
            composed = compose_state(str(state["label"]),
                                     [last_fragments[label] for label in labels],
                                     energy, state_directory)
            checkpoint.update(composed)
            _atomic_json(checkpoint_path, checkpoint)
            return checkpoint
        previous_energy = energy
    raise WorkflowError(f"state {state['label']} did not converge in {maximum_cycles} cycles")


def write_postprocess_inputs(spec: Mapping[str, object],
                             geometry: Mapping[str, object],
                             results: Sequence[Mapping[str, object]],
                             output_directory: Path) -> Path:
    first_fragment = read_fragment(Path(next(iter(results[0]["fragments"].values()))))
    overlap = list(first_fragment["ao_overlap"])
    dimension = int(first_fragment["ao_dimension"])
    overlap_path = output_directory / "ao_overlap.fde_matrix"
    overlap_path.write_text("FDE_AO_MATRIX 1\n" + f"DIMENSION {dimension}\nVALUES {len(overlap)} "
                            + " ".join(format(value, ".17g") for value in overlap)
                            + "\nEND\n", encoding="utf-8")
    fragments = list(spec["fragments"])
    controls = dict(spec.get("controls", {}))
    lines = ["FDE_CONFIG 1", f"ATOM_COUNT {sum(len(f['atom_indices']) for f in fragments)}"]
    for fragment in fragments:
        lines.append(f"FRAGMENT {fragment['label']} {fragment['neutral_valence_electrons']} "
                     f"{len(fragment['atom_indices'])} "
                     + " ".join(str(value) for value in fragment["atom_indices"]))
    for state in spec["states"]:
        assignments: List[str] = []
        for fragment in fragments:
            label = fragment["label"]
            assignments.extend((label, str(state["fragments"][label]["charge"]),
                                str(state["fragments"][label]["spin"])))
        lines.append(f"STATE {state['label']} {state['total_charge']} {state['total_spin']} "
                     f"{len(fragments)} {' '.join(assignments)}")
    for result in results:
        lines.append(f"DETERMINANT {result['state']} {_absolute_token(Path(result['determinant']), 'determinant path')}")
        lines.append(f"DIAGONAL_ENERGY_RY {result['state']} {float(result['energy_ry']):.17g}")
        lines.append(f"LINEARIZED_STATE {result['state']} "
                     f"{_absolute_token(Path(result['linearized_state']), 'linearized-state path')}")
    labels = [state["label"] for state in spec["states"]]
    fragment_labels = [fragment["label"] for fragment in fragments]
    lines.extend((f"AO_OVERLAP {_absolute_token(overlap_path, 'AO overlap path')}",
                  "OUTPUT_PREFIX fde_diabatic",
                  f"KEDF {controls.get('kedf', 'lc94')}", "DENSITY_FLOOR_BOHR3 1e-12",
                  "MAX_SCF_ITERATIONS 100", "SCF_DENSITY_TOLERANCE 1e-8",
                  "ELECTRON_TOLERANCE 1e-8", "MIXING_BETA 0.3",
                  "MAX_FREEZE_THAW_CYCLES 20", "FREEZE_THAW_DENSITY_TOLERANCE 1e-7",
                  "ENERGY_TOLERANCE_RY 1e-8",
                  f"UPDATE_ORDER {len(fragment_labels)} {' '.join(fragment_labels)}",
                  f"K_STATES {len(labels)} {' '.join(labels)}",
                  f"L_FRAGMENTS {len(fragment_labels)} {' '.join(fragment_labels)}",
                  f"M_FRAGMENTS {len(fragment_labels)} {' '.join(fragment_labels)}",
                  f"SINGULAR_VALUE_TOLERANCE {controls.get('singular_value_tolerance', 1e-10)}",
                  f"OVERLAP_EIGENVALUE_CUTOFF {controls.get('overlap_eigenvalue_cutoff', 1e-9)}",
                  f"SYMMETRY_TOLERANCE {controls.get('symmetry_tolerance', 1e-10)}",
                  f"RESIDUAL_TOLERANCE {controls.get('residual_tolerance', 1e-8)}",
                  f"MINIMUM_ROOT_OVERLAP {controls.get('minimum_root_overlap', 0.5)}",
                  "CALCULATE_FORCE false", "HARTREE_RECIPROCITY_TOLERANCE_RY 1e-8",
                  "END_FDE_CONFIG"))
    path = output_directory / "FDE_POSTPROCESS_CONFIG"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_workflow(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_spec(spec)
    geometries = spec.get("geometries")
    if not geometries:
        geometries = [{"label": "geometry-000", "template_directory": spec["template_directory"],
                       "initial_densities": spec.get("initial_densities", {})}]
    root = Path(spec["work_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pes_rows: List[Dict[str, object]] = []
    for geometry in geometries:
        geometry = dict(geometry)
        geometry["label"] = _token(geometry["label"], "geometry label")
        geometry["template_directory"] = str(Path(geometry["template_directory"]).resolve())
        geometry_directory = root / geometry["label"]
        geometry_directory.mkdir(parents=True, exist_ok=True)
        results = [run_state(spec, geometry, state, geometry_directory) for state in spec["states"]]
        postprocess = write_postprocess_inputs(spec, geometry, results, geometry_directory)
        pes_rows.append({"geometry": geometry["label"],
                         "states": {result["state"]: result["energy_ry"] for result in results},
                         "postprocess_config": str(postprocess.resolve())})
    _atomic_json(root / "fde_pes.json", {"schema_version": 1, "points": pes_rows})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("spec", type=Path)
    arguments = parser.parse_args(argv)
    try:
        spec = json.loads(arguments.spec.read_text(encoding="utf-8"))
        validate_spec(spec)
        if arguments.command == "run":
            run_workflow(arguments.spec)
    except (OSError, ValueError, KeyError, TypeError, WorkflowError) as error:
        print(f"fde_workflow: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
