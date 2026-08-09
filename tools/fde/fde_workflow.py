#!/usr/bin/env python3
"""Restartable two-fragment FDE freeze--thaw and PES workflow for ABACUS."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
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


def _label(value: object, description: str) -> str:
    text = _token(value, description)
    if text in (".", "..") or re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", text) is None:
        raise WorkflowError(
            f"{description} must use only letters, digits, dot, underscore, or hyphen")
    return text


def spin_population(neutral_electrons: int, charge: int, spin: int) -> Tuple[int, int]:
    electrons = neutral_electrons - charge
    if electrons < 0 or abs(spin) > electrons or (electrons + spin) % 2:
        raise WorkflowError("fragment charge/spin assignment has an invalid electron parity")
    return (electrons + spin) // 2, (electrons - spin) // 2


def fragment_mixing_parameters(controls: Mapping[str, object],
                               fragment_label: str) -> Dict[str, object]:
    fragment_mixing = controls.get("fragment_mixing", {})
    if not isinstance(fragment_mixing, dict):
        raise WorkflowError("fragment_mixing must be a JSON object")
    settings = fragment_mixing.get(fragment_label, {})
    if not isinstance(settings, dict):
        raise WorkflowError(
            f"fragment_mixing entry for {fragment_label} must be a JSON object")
    return {name: settings[name]
            for name in ("mixing_type", "mixing_beta", "mixing_beta_mag")
            if name in settings}


def validate_spec(spec: Mapping[str, object]) -> None:
    if int(spec.get("schema_version", 0)) != 1:
        raise WorkflowError("workflow schema_version must be 1")
    command = spec.get("abacus_command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise WorkflowError("abacus_command must be a nonempty JSON string array")
    controls = spec.get("controls", {})
    if not isinstance(controls, dict):
        raise WorkflowError("controls must be a JSON object")
    solver = _token(controls.get("ks_solver", "lapack"), "ks_solver")
    if solver not in ("lapack", "genelpa", "elpa", "scalapack_gvx"):
        raise WorkflowError(
            "ks_solver must be lapack, genelpa, elpa, or scalapack_gvx")
    if int(controls.get("kpar", 1)) != 1:
        raise WorkflowError("the Gamma-point FDE workflow currently requires kpar 1")
    retained_cycles = controls.get("retain_completed_cycles", 0)
    if (isinstance(retained_cycles, bool)
            or not isinstance(retained_cycles, int)
            or retained_cycles < 0):
        raise WorkflowError("retain_completed_cycles must be a nonnegative integer")
    remove_restarts = controls.get("remove_abacus_restart_files", False)
    if not isinstance(remove_restarts, bool):
        raise WorkflowError("remove_abacus_restart_files must be a boolean")
    allow_partial_scf = controls.get("allow_partial_scf", False)
    if not isinstance(allow_partial_scf, bool):
        raise WorkflowError("allow_partial_scf must be a boolean")
    integer_controls = {
        "maximum_freeze_thaw_cycles": (20, 1),
        "maximum_scf_iterations": (100, 1),
        "inexact_freeze_thaw_cycles": (0, 0),
        "inexact_scf_iterations": (50, 1),
        "strict_confirmation_cycles": (2, 2),
    }
    validated_integers: Dict[str, int] = {}
    for name, (default, minimum) in integer_controls.items():
        value = controls.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise WorkflowError(f"{name} must be an integer not smaller than {minimum}")
        validated_integers[name] = value
    inexact_cycles = validated_integers["inexact_freeze_thaw_cycles"]
    if inexact_cycles > 0 and not allow_partial_scf:
        raise WorkflowError("inexact freeze-thaw cycles require allow_partial_scf=true")
    required_cycles = inexact_cycles + validated_integers["strict_confirmation_cycles"]
    if validated_integers["maximum_freeze_thaw_cycles"] < required_cycles:
        raise WorkflowError(
            "maximum_freeze_thaw_cycles cannot fit the inexact and strict-confirmation stages")
    for name, default in (("scf_density_tolerance", 1e-8),
                          ("inexact_scf_density_tolerance", 1e-3)):
        value = controls.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) <= 0.0):
            raise WorkflowError(f"{name} must be finite and positive")
    fragments = spec.get("fragments")
    states = spec.get("states")
    if not isinstance(fragments, list) or len(fragments) != 2:
        raise WorkflowError("the production workflow currently requires exactly two fragments")
    if not isinstance(states, list) or len(states) < 2:
        raise WorkflowError("the workflow requires at least two diabatic states")
    labels = [_label(fragment["label"], "fragment label") for fragment in fragments]
    if len(set(labels)) != len(labels):
        raise WorkflowError("fragment labels must be unique")
    fragment_mixing = controls.get("fragment_mixing", {})
    if not isinstance(fragment_mixing, dict):
        raise WorkflowError("fragment_mixing must be a JSON object")
    if not set(fragment_mixing).issubset(labels):
        raise WorkflowError("fragment_mixing contains an unknown fragment label")
    for label in fragment_mixing:
        settings = fragment_mixing_parameters(controls, label)
        mixing_type = _token(settings.get("mixing_type", "broyden"),
                             f"mixing_type for fragment {label}")
        if mixing_type not in ("plain", "pulay", "broyden"):
            raise WorkflowError(
                f"mixing_type for fragment {label} must be plain, pulay, or broyden")
        for name in ("mixing_beta", "mixing_beta_mag"):
            if name not in settings:
                continue
            value = settings[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) <= 0.0):
                raise WorkflowError(
                    f"{name} for fragment {label} must be finite and positive")
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
        state_labels.append(_label(state["label"], "state label"))
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
        if "FDE_UNIFORM_DENSITY_SEED" in records:
            schema_version = int(records["FDE_UNIFORM_DENSITY_SEED"][0][0])
            grid = records["GRID"][0]
            grid_size = int(grid[0]) * int(grid[1]) * int(grid[2])
            uniform = records["RHO_UNIFORM"][0]
            alpha = [float(uniform[0])] * grid_size
            beta = [float(uniform[1])] * grid_size
            cycle = 0
            scf_converged = False
            scf_iterations = 0
            scf_density_residual = None
            initialization_seed = True
        else:
            schema_version = int(records["FDE_DENSITY_ARTIFACT"][0][0])
            alpha = [float(value) for value in records["RHO_ALPHA"][0][1:]]
            beta = [float(value) for value in records["RHO_BETA"][0][1:]]
            if (int(records["RHO_ALPHA"][0][0]) != len(alpha)
                    or int(records["RHO_BETA"][0][0]) != len(beta)):
                raise WorkflowError("density vector length is inconsistent")
            grid = records["GRID"][0]
            scf = records["SCF"][0]
            cycle = int(scf[0])
            convergence_flag = int(scf[1])
            if convergence_flag not in (0, 1):
                raise WorkflowError("density SCF convergence flag must be zero or one")
            scf_converged = convergence_flag == 1
            if schema_version >= 2:
                scf_iterations = int(scf[2])
                scf_density_residual = float(scf[3])
            else:
                scf_iterations = 0
                scf_density_residual = None
            initialization_seed = False
        if not alpha or len(alpha) != len(beta):
            raise WorkflowError("density grid must be nonempty and spin-compatible")
        grid = records["GRID"][0]
        return {
            "fragment": records["FRAGMENT"][0][0],
            "state": records["STATE"][0][0],
            "geometry": records["GEOMETRY"][0][0],
            "schema_version": schema_version,
            "cycle": cycle,
            "scf_converged": scf_converged,
            "scf_iterations": scf_iterations,
            "scf_density_residual": scf_density_residual,
            "initialization_seed": initialization_seed,
            "grid_size": len(alpha),
            "cell_volume": float(grid[3]),
            "alpha": alpha,
            "beta": beta,
        }
    except (KeyError, IndexError, ValueError) as error:
        raise WorkflowError(f"malformed density artifact {path}: {error}") from error


def select_scf_density(job_directory: Path,
                       allow_partial_scf: bool) -> Tuple[Path, Dict[str, object]]:
    converged_path = job_directory / "result.fde_density"
    partial_path = job_directory / "result.partial.fde_density"
    if converged_path.is_file() and partial_path.is_file():
        raise WorkflowError(f"ABACUS produced conflicting FDE densities in {job_directory}")
    if converged_path.is_file():
        density = read_density(converged_path)
        if not density["scf_converged"]:
            raise WorkflowError(f"converged FDE density has a false SCF flag: {converged_path}")
        return converged_path, density
    if not partial_path.is_file():
        raise WorkflowError(f"ABACUS did not produce an FDE density in {job_directory}")
    if not allow_partial_scf:
        raise WorkflowError(
            f"ABACUS produced only a partial FDE density in {job_directory}; "
            "set controls.allow_partial_scf=true to accept it")
    density = read_density(partial_path)
    residual = density["scf_density_residual"]
    if (density["schema_version"] < 2 or density["scf_converged"]
            or density["scf_iterations"] <= 0 or residual is None
            or not math.isfinite(float(residual)) or float(residual) < 0.0):
        raise WorkflowError(f"partial FDE density lacks valid SCF metadata: {partial_path}")
    return partial_path, density


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
        elif key == "SCF_CONVERGED":
            flag = int(fields[1])
            if flag not in (0, 1):
                raise WorkflowError(f"invalid SCF convergence flag in {path}")
            result["scf_converged"] = flag == 1
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
    if result.get("scf_converged") is not True:
        raise WorkflowError(f"FDE fragment is not a converged SCF result: {path}")
    if dimension <= 0 or any(len(orbital["coefficients"]) != dimension
                             for name in ("alpha", "beta") for orbital in result[name]):
        raise WorkflowError(f"invalid occupied-orbital dimensions in {path}")
    return result


def canonical_two_fragment_energy(artifacts: Sequence[Mapping[str, object]],
                                  tolerance: float) -> float:
    if len(artifacts) != 2:
        raise WorkflowError("canonical runtime energy currently requires two fragment artifacts")
    if not all(item.get("scf_converged") is True for item in artifacts):
        raise WorkflowError("canonical runtime energy requires converged fragment SCFs")
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


def scf_schedule(controls: Mapping[str, object], cycle: int) -> Dict[str, object]:
    if cycle <= 0:
        raise WorkflowError("freeze-thaw cycle must be positive")
    inexact_cycles = int(controls.get("inexact_freeze_thaw_cycles", 0))
    strict = cycle > inexact_cycles
    if strict:
        return {
            "mode": "strict",
            "strict": True,
            "maximum_iterations": int(controls.get("maximum_scf_iterations", 100)),
            "density_tolerance": float(controls.get("scf_density_tolerance", 1e-8)),
        }
    return {
        "mode": "inexact",
        "strict": False,
        "maximum_iterations": int(controls.get("inexact_scf_iterations", 50)),
        "density_tolerance": float(
            controls.get("inexact_scf_density_tolerance", 1e-3)),
    }


def write_runtime_config(path: Path,
                         spec: Mapping[str, object],
                         state: Mapping[str, object],
                         active_label: str,
                         densities: Mapping[str, Path],
                         output_prefix: str,
                         maximum_scf_iterations: int,
                         scf_density_tolerance: float) -> None:
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
                  f"MAX_SCF_ITERATIONS {maximum_scf_iterations}",
                  f"SCF_DENSITY_TOLERANCE {scf_density_tolerance}",
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


def remove_abacus_restart_files(job_directory: Path) -> None:
    """Remove large, reproducible charge restart files from one finished job."""
    for output_directory in job_directory.glob("OUT.*"):
        if not output_directory.is_dir() or output_directory.is_symlink():
            continue
        for restart in output_directory.glob("*-CHARGE-DENSITY.restart"):
            if restart.is_file() and not restart.is_symlink():
                restart.unlink()


def prune_completed_cycles(state_directory: Path,
                           current_cycle: int,
                           retain_completed_cycles: int) -> None:
    """Keep only the requested newest complete freeze--thaw cycle directories."""
    if retain_completed_cycles <= 0:
        return
    first_retained_cycle = current_cycle - retain_completed_cycles + 1
    for candidate in state_directory.iterdir():
        match = re.fullmatch(r"cycle-([0-9]+)", candidate.name)
        if (match is not None
                and int(match.group(1)) < first_retained_cycle
                and candidate.is_dir()
                and not candidate.is_symlink()):
            shutil.rmtree(candidate)


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
    ks_solver = str(controls.get("ks_solver", "lapack"))
    kpar = int(controls.get("kpar", 1))
    retain_completed_cycles = int(controls.get("retain_completed_cycles", 0))
    remove_restarts = bool(controls.get("remove_abacus_restart_files", False))
    allow_partial_scf = bool(controls.get("allow_partial_scf", False))
    required_strict_confirmations = int(controls.get("strict_confirmation_cycles", 2))
    state_directory = output_directory / str(state["label"])
    state_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_directory / "checkpoint.json"
    initial = geometry.get("initial_densities", {}).get(state["label"], state.get("initial_densities", {}))
    if set(initial) != set(labels):
        raise WorkflowError(f"state {state['label']} requires one initial density per fragment")
    densities = {label: Path(initial[label]).resolve() for label in labels}
    start_cycle = 1
    previous_complete_energy = None
    previous_cycle_complete = False
    strict_confirmations = 0
    history: List[Dict[str, object]] = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("converged"):
            return checkpoint
        densities = {label: Path(checkpoint["densities"][label]) for label in labels}
        start_cycle = int(checkpoint["cycle"]) + 1
        previous_cycle_complete = bool(checkpoint.get("strict_cycle_complete", False))
        if previous_cycle_complete:
            previous_complete_energy = checkpoint.get("energy_ry")
        history = list(checkpoint.get("history", []))
        strict_confirmations = int(checkpoint.get("strict_confirmations", 0))

    neutral = {fragment["label"]: int(fragment["neutral_valence_electrons"])
               for fragment in fragments}
    for cycle in range(start_cycle, maximum_cycles + 1):
        schedule = scf_schedule(controls, cycle)
        strict_scf = bool(schedule["strict"])
        old_density_data = {label: read_density(path) for label, path in densities.items()}
        cycle_fragments: List[Mapping[str, object]] = []
        cycle_fragment_paths: Dict[str, Path] = {}
        cycle_scf: Dict[str, Dict[str, object]] = {}
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
            write_runtime_config(config_path,
                                 spec,
                                 state,
                                 active_label,
                                 densities,
                                 "result",
                                 int(schedule["maximum_iterations"]),
                                 float(schedule["density_tolerance"]))
            input_parameters: Dict[str, object] = {
                "calculation": "scf", "basis_type": "lcao", "gamma_only": 1,
                "nspin": 2, "noncolin": 0, "lspinorb": 0, "symmetry": 0,
                "dft_functional": "pbe", "ks_solver": ks_solver, "kpar": kpar,
                "nelec": alpha + beta, "nupdown": alpha - beta,
                "fde_task": "embedded_scf", "fde_config": "FDE_CONFIG",
                "scf_nmax": int(schedule["maximum_iterations"]),
                "scf_thr": float(schedule["density_tolerance"]),
            }
            input_parameters.update(
                fragment_mixing_parameters(controls, active_label))
            patch_input(job_directory / "INPUT", input_parameters)
            log_path = job_directory / "fde_abacus.log"
            environment = dict(os.environ)
            environment.setdefault("OMP_NUM_THREADS", "1")
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(list(spec["abacus_command"]), cwd=job_directory,
                                           env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                raise WorkflowError(f"ABACUS failed in {job_directory}; see {log_path}")
            density_path, density_data = select_scf_density(job_directory,
                                                            allow_partial_scf)
            inner_converged = bool(density_data["scf_converged"])
            fragment_path = job_directory / "result.fde_fragment"
            if inner_converged and not fragment_path.is_file():
                raise WorkflowError(
                    f"converged ABACUS job did not produce an FDE fragment in {job_directory}")
            if not inner_converged and fragment_path.exists():
                raise WorkflowError(
                    f"partial ABACUS job produced a final FDE fragment in {job_directory}")
            if remove_restarts:
                remove_abacus_restart_files(job_directory)
            densities[active_label] = density_path.resolve()
            cycle_scf[active_label] = {
                "converged": inner_converged,
                "iterations": density_data["scf_iterations"],
                "density_residual": density_data["scf_density_residual"],
            }
            if inner_converged:
                cycle_fragment_paths[active_label] = fragment_path.resolve()
                cycle_fragments.append(read_fragment(fragment_path))
        residual = max(density_rms(old_density_data[label], read_density(densities[label]))
                       for label in labels)
        all_inner_converged = len(cycle_fragment_paths) == len(labels)
        strict_cycle_complete = strict_scf and all_inner_converged
        energy = (canonical_two_fragment_energy(cycle_fragments, symmetry_tolerance)
                  if strict_cycle_complete else None)
        energy_change = (abs(float(energy) - float(previous_complete_energy))
                         if (energy is not None and previous_cycle_complete
                             and previous_complete_energy is not None) else None)
        strict_confirmations = (strict_confirmations + 1
                                if strict_cycle_complete else 0)
        converged = (strict_cycle_complete
                     and strict_confirmations >= required_strict_confirmations
                     and residual <= density_tolerance
                     and energy_change is not None
                     and energy_change <= energy_tolerance)
        history.append({"cycle": cycle, "density_rms": residual, "energy_ry": energy,
                        "energy_change_ry": energy_change,
                        "scf_mode": schedule["mode"],
                        "all_inner_scf_converged": all_inner_converged,
                        "strict_cycle_complete": strict_cycle_complete,
                        "strict_confirmations": strict_confirmations,
                        "inner_scf": cycle_scf})
        checkpoint: Dict[str, object] = {
            "schema_version": 1, "geometry": geometry["label"], "state": state["label"],
            "cycle": cycle, "converged": converged, "density_rms": residual,
            "energy_ry": energy, "energy_change_ry": energy_change,
            "scf_mode": schedule["mode"],
            "all_inner_scf_converged": all_inner_converged,
            "strict_cycle_complete": strict_cycle_complete,
            "strict_confirmations": strict_confirmations,
            "inner_scf": cycle_scf,
            "densities": {label: str(densities[label]) for label in labels},
            "fragments": {label: str(path) for label, path in cycle_fragment_paths.items()},
            "history": history,
        }
        _atomic_json(checkpoint_path, checkpoint)
        prune_completed_cycles(state_directory, cycle, retain_completed_cycles)
        if converged:
            composed = compose_state(str(state["label"]),
                                     [cycle_fragment_paths[label] for label in labels],
                                     float(energy), state_directory)
            checkpoint.update(composed)
            _atomic_json(checkpoint_path, checkpoint)
            return checkpoint
        previous_cycle_complete = strict_cycle_complete
        previous_complete_energy = energy if strict_cycle_complete else None
    raise WorkflowError(f"state {state['label']} did not converge in {maximum_cycles} cycles")


def write_postprocess_inputs(spec: Mapping[str, object],
                             geometry: Mapping[str, object],
                             results: Sequence[Mapping[str, object]],
                             output_directory: Path) -> Path:
    fragment_labels = {str(fragment["label"]) for fragment in spec["fragments"]}
    for result in results:
        if (result.get("converged") is not True
                or set(result.get("fragments", {})) != fragment_labels):
            raise WorkflowError(
                "diabatic postprocessing requires converged artifacts for every fragment")
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
        geometry["label"] = _label(geometry["label"], "geometry label")
        geometry["template_directory"] = str(Path(geometry["template_directory"]).resolve())
        geometry_directory = root / geometry["label"]
        geometry_directory.mkdir(parents=True, exist_ok=True)
        results = [run_state(spec, geometry, state, geometry_directory) for state in spec["states"]]
        postprocess = write_postprocess_inputs(spec, geometry, results, geometry_directory)
        postprocess_directory = geometry_directory / "postprocess"
        postprocess_directory.mkdir(exist_ok=True)
        postprocess_input = postprocess_directory / "INPUT"
        postprocess_input.write_text(
            "INPUT_PARAMETERS\ncalculation              scf\n"
            "fde_task                diabatic_postprocess\n"
            f"fde_config              {_absolute_token(postprocess, 'postprocess config')}\n",
            encoding="utf-8")
        if bool(spec.get("run_postprocess", True)):
            postprocess_command = spec.get("postprocess_command", spec["abacus_command"])
            if not isinstance(postprocess_command, list) or not postprocess_command:
                raise WorkflowError("postprocess_command must be a nonempty string array")
            environment = dict(os.environ)
            environment.setdefault("OMP_NUM_THREADS", "1")
            log_path = postprocess_directory / "fde_postprocess.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(list(postprocess_command), cwd=postprocess_directory,
                                           env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                raise WorkflowError(f"FDE postprocess failed; see {log_path}")
        row: Dict[str, object] = {
            "geometry": geometry["label"],
            "coordinate_angstrom": geometry.get("coordinate_angstrom"),
            "states": {result["state"]: result["energy_ry"] for result in results},
            "postprocess_config": str(postprocess.resolve()),
            "postprocess_directory": str(postprocess_directory.resolve()),
        }
        result_path = postprocess_directory / "fde_diabatic.fde_diabatic"
        if result_path.is_file():
            records = _records(result_path)
            row["pairs"] = [{"first_state_index": int(fields[0]),
                             "second_state_index": int(fields[1]),
                             "overlap": float(fields[2]),
                             "h12_ry": float(fields[3]),
                             "orthogonalized_coupling_ry": float(fields[4])}
                            for fields in records.get("PAIR", [])]
            adiabatic = records.get("ADIABATIC_ENERGIES_RY", [["0"]])[0]
            row["adiabatic_energies_ry"] = [float(value) for value in adiabatic[1:]]
        pes_rows.append(row)
    _atomic_json(root / "fde_pes.json", {"schema_version": 1, "points": pes_rows})
    state_labels = [state["label"] for state in spec["states"]]
    table_lines = ["geometry\tcoordinate_angstrom\t" + "\t".join(
        f"{label}_energy_ry" for label in state_labels)
        + "\toverlap\th12_ry\torthogonalized_coupling_ry"]
    for row in pes_rows:
        pair = row.get("pairs", [{}])[0] if row.get("pairs") else {}
        coordinate = row.get("coordinate_angstrom")
        fields = [str(row["geometry"]), "" if coordinate is None else format(float(coordinate), ".17g")]
        fields.extend(format(float(row["states"][label]), ".17g") for label in state_labels)
        fields.extend("" if key not in pair else format(float(pair[key]), ".17g")
                      for key in ("overlap", "h12_ry", "orthogonalized_coupling_ry"))
        table_lines.append("\t".join(fields))
    (root / "fde_pes.tsv").write_text("\n".join(table_lines) + "\n", encoding="utf-8")


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
