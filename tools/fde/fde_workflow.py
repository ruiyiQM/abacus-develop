#!/usr/bin/env python3
"""Restartable two-fragment FDE freeze--thaw and PES workflow for ABACUS."""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
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


def embedded_scf_spin_parameters(controls: Mapping[str, object],
                                 neutral_electrons: int,
                                 charge: int,
                                 spin: int) -> Dict[str, int]:
    """Map a fragment assignment to ABACUS RKS/UKS input parameters."""
    alpha, beta = spin_population(neutral_electrons, charge, spin)
    spin_mode = _token(controls.get("spin_mode", "uks"), "spin_mode").lower()
    if spin_mode not in ("rks", "uks"):
        raise WorkflowError("spin_mode must be rks or uks")
    if spin_mode == "rks":
        if alpha != beta:
            raise WorkflowError(
                "RKS FDE requires every fragment to have an even electron count "
                "and spin 0; use spin_mode=uks for odd-electron fragments")
        return {"nspin": 1, "nelec": alpha + beta, "nupdown": 0}
    return {"nspin": 2, "nelec": alpha + beta, "nupdown": alpha - beta}


MIXING_PARAMETER_NAMES = (
    "mixing_type", "mixing_beta", "mixing_beta_mag", "mixing_ndim",
    "mixing_restart", "mixing_dmr", "mixing_gg0", "mixing_gg0_mag",
    "mixing_gg0_min",
)


def _mixing_settings(value: object, description: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise WorkflowError(f"{description} must be a JSON object")
    unknown = set(value) - set(MIXING_PARAMETER_NAMES)
    if unknown:
        raise WorkflowError(
            f"{description} contains unsupported keys: {', '.join(sorted(unknown))}")
    return {name: value[name] for name in MIXING_PARAMETER_NAMES if name in value}


def _validate_mixing_settings(settings: Mapping[str, object], description: str) -> None:
    if "mixing_type" in settings:
        mixing_type = _token(settings["mixing_type"], f"mixing_type in {description}")
        if mixing_type not in ("plain", "pulay", "broyden"):
            raise WorkflowError(
                f"mixing_type in {description} must be plain, pulay, or broyden")
    positive = ("mixing_beta", "mixing_beta_mag")
    nonnegative = ("mixing_restart", "mixing_gg0", "mixing_gg0_mag",
                   "mixing_gg0_min")
    for name in positive + nonnegative:
        if name not in settings:
            continue
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkflowError(f"{name} in {description} must be numeric")
        minimum_ok = float(value) > 0.0 if name in positive else float(value) >= 0.0
        if not math.isfinite(float(value)) or not minimum_ok:
            qualifier = "positive" if name in positive else "nonnegative"
            raise WorkflowError(f"{name} in {description} must be finite and {qualifier}")
    if "mixing_ndim" in settings:
        value = settings["mixing_ndim"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorkflowError(f"mixing_ndim in {description} must be a positive integer")
    if "mixing_dmr" in settings and not isinstance(settings["mixing_dmr"], bool):
        raise WorkflowError(f"mixing_dmr in {description} must be a boolean")


def fragment_mixing_parameters(controls: Mapping[str, object],
                               fragment_label: str,
                               stage: Mapping[str, object] | None = None,
                               retry: int = 0) -> Dict[str, object]:
    settings = {name: controls[name]
                for name in MIXING_PARAMETER_NAMES if name in controls}
    fragment_mixing = controls.get("fragment_mixing", {})
    if not isinstance(fragment_mixing, dict):
        raise WorkflowError("fragment_mixing must be a JSON object")
    fragment_settings = _mixing_settings(
        fragment_mixing.get(fragment_label, {}),
        f"fragment_mixing entry for {fragment_label}")
    settings.update(fragment_settings)
    if stage is not None:
        settings.update(_mixing_settings(
            stage.get("mixing", {}),
            f"mixing for adaptive SCF stage {stage.get('name', '<unnamed>')}"))
        stage_fragments = stage.get("fragment_mixing", {})
        if not isinstance(stage_fragments, dict):
            raise WorkflowError("adaptive SCF stage fragment_mixing must be a JSON object")
        settings.update(_mixing_settings(
            stage_fragments.get(fragment_label, {}),
            f"adaptive SCF stage mixing for fragment {fragment_label}"))
    if retry > 0:
        recovery = controls.get("mixing_recovery", {})
        fallbacks = recovery.get("fallbacks", []) if isinstance(recovery, dict) else []
        if retry > len(fallbacks):
            raise WorkflowError("mixing recovery retry exceeds the fallback list")
        settings.update(_mixing_settings(
            fallbacks[retry - 1], f"mixing recovery fallback {retry}"))
    return settings


def adaptive_scf_stages(controls: Mapping[str, object]) -> List[Dict[str, object]]:
    adaptive = controls.get("adaptive_scf", {})
    if not isinstance(adaptive, dict):
        raise WorkflowError("adaptive_scf must be a JSON object")
    stages = adaptive.get("stages", [])
    if not isinstance(stages, list):
        raise WorkflowError("adaptive_scf.stages must be a JSON array")
    return [dict(stage) if isinstance(stage, dict) else stage for stage in stages]


def validate_spec(spec: Mapping[str, object]) -> None:
    if int(spec.get("schema_version", 0)) != 1:
        raise WorkflowError("workflow schema_version must be 1")
    command = spec.get("abacus_command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise WorkflowError("abacus_command must be a nonempty JSON string array")
    controls = spec.get("controls", {})
    if not isinstance(controls, dict):
        raise WorkflowError("controls must be a JSON object")
    kedf = _token(controls.get("kedf", "lc94"), "kedf").lower()
    if kedf not in ("lc94", "thomas_fermi", "tf"):
        raise WorkflowError("kedf must be lc94, thomas_fermi, or tf")
    solver = _token(controls.get("ks_solver", "lapack"), "ks_solver")
    if solver not in ("lapack", "genelpa", "elpa", "scalapack_gvx"):
        raise WorkflowError(
            "ks_solver must be lapack, genelpa, elpa, or scalapack_gvx")
    spin_mode = _token(controls.get("spin_mode", "uks"), "spin_mode").lower()
    if spin_mode not in ("rks", "uks"):
        raise WorkflowError("spin_mode must be rks or uks")
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
    adaptive = controls.get("adaptive_scf", {})
    if not isinstance(adaptive, dict):
        raise WorkflowError("adaptive_scf must be a JSON object")
    adaptive_enabled = adaptive.get("enabled", False)
    if not isinstance(adaptive_enabled, bool):
        raise WorkflowError("adaptive_scf.enabled must be a boolean")
    if adaptive_enabled and inexact_cycles > 0:
        raise WorkflowError(
            "adaptive_scf cannot be combined with inexact_freeze_thaw_cycles")
    if (inexact_cycles > 0 or adaptive_enabled) and not allow_partial_scf:
        raise WorkflowError(
            "inexact or adaptive freeze-thaw stages require allow_partial_scf=true")
    required_cycles = inexact_cycles + validated_integers["strict_confirmation_cycles"]
    if (not adaptive_enabled
            and validated_integers["maximum_freeze_thaw_cycles"] < required_cycles):
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
    update_order = controls.get("update_order", labels)
    if (not isinstance(update_order, list) or len(update_order) != len(labels)
            or not all(isinstance(label, str) for label in update_order)
            or set(update_order) != set(labels)):
        raise WorkflowError("update_order must be a permutation of fragment labels")
    update_scheme = _token(
        controls.get("update_scheme", "auto"), "update_scheme").lower()
    if update_scheme not in ("auto", "gauss_seidel", "jacobi"):
        raise WorkflowError("update_scheme must be auto, gauss_seidel, or jacobi")
    parallelism = controls.get("jacobi_parallelism", 1)
    if (isinstance(parallelism, bool) or not isinstance(parallelism, int)
            or parallelism < 1 or parallelism > len(labels)):
        raise WorkflowError("jacobi_parallelism must be between 1 and the fragment count")
    if parallelism > 1 and update_scheme != "jacobi":
        raise WorkflowError("jacobi_parallelism greater than one requires update_scheme=jacobi")

    outer_mixing = controls.get("outer_mixing", {})
    if not isinstance(outer_mixing, dict):
        raise WorkflowError("outer_mixing must be a JSON object")
    unknown_outer = set(outer_mixing) - {
        "type", "beta", "history", "regularization", "apply_in_strict",
    }
    if unknown_outer:
        raise WorkflowError("outer_mixing contains unsupported keys")
    outer_type = _token(outer_mixing.get("type", "none"), "outer_mixing.type")
    if outer_type not in ("none", "linear", "anderson"):
        raise WorkflowError("outer_mixing.type must be none, linear, or anderson")
    outer_beta = outer_mixing.get("beta", 0.5)
    if (isinstance(outer_beta, bool) or not isinstance(outer_beta, (int, float))
            or not math.isfinite(float(outer_beta))
            or float(outer_beta) <= 0.0 or float(outer_beta) > 1.0):
        raise WorkflowError("outer_mixing.beta must be in (0, 1]")
    outer_history = outer_mixing.get("history", 4)
    if (isinstance(outer_history, bool) or not isinstance(outer_history, int)
            or outer_history < 1 or outer_history > 8
            or (outer_type == "anderson" and outer_history < 2)):
        raise WorkflowError("outer_mixing.history must be 2--8 for Anderson")
    regularization = outer_mixing.get("regularization", 1e-10)
    if (isinstance(regularization, bool)
            or not isinstance(regularization, (int, float))
            or not math.isfinite(float(regularization))
            or float(regularization) <= 0.0):
        raise WorkflowError("outer_mixing.regularization must be finite and positive")
    apply_in_strict = outer_mixing.get("apply_in_strict", False)
    if not isinstance(apply_in_strict, bool) or apply_in_strict:
        raise WorkflowError(
            "outer_mixing.apply_in_strict must currently be false so final artifacts agree")
    fragment_mixing = controls.get("fragment_mixing", {})
    if not isinstance(fragment_mixing, dict):
        raise WorkflowError("fragment_mixing must be a JSON object")
    if not set(fragment_mixing).issubset(labels):
        raise WorkflowError("fragment_mixing contains an unknown fragment label")
    _validate_mixing_settings(
        {name: controls[name] for name in MIXING_PARAMETER_NAMES if name in controls},
        "global controls")
    for label in fragment_mixing:
        settings = _mixing_settings(
            fragment_mixing[label], f"fragment_mixing entry for {label}")
        _validate_mixing_settings(settings, f"fragment {label}")

    recovery = controls.get("mixing_recovery", {})
    if not isinstance(recovery, dict):
        raise WorkflowError("mixing_recovery must be a JSON object")
    unknown_recovery = set(recovery) - {"enabled", "fallbacks"}
    if unknown_recovery:
        raise WorkflowError("mixing_recovery contains unsupported keys")
    recovery_enabled = recovery.get("enabled", False)
    if not isinstance(recovery_enabled, bool):
        raise WorkflowError("mixing_recovery.enabled must be a boolean")
    fallbacks = recovery.get("fallbacks", [])
    if not isinstance(fallbacks, list):
        raise WorkflowError("mixing_recovery.fallbacks must be a JSON array")
    if recovery_enabled and not fallbacks:
        raise WorkflowError("enabled mixing_recovery requires at least one fallback")
    for index, fallback in enumerate(fallbacks, 1):
        settings = _mixing_settings(fallback, f"mixing recovery fallback {index}")
        _validate_mixing_settings(settings, f"mixing recovery fallback {index}")

    if adaptive_enabled:
        unknown_adaptive = set(adaptive) - {"enabled", "force_strict_cycle", "stages"}
        if unknown_adaptive:
            raise WorkflowError("adaptive_scf contains unsupported keys")
        stages = adaptive_scf_stages(controls)
        if len(stages) < 2:
            raise WorkflowError("adaptive_scf requires at least two stages")
        previous_minimum = math.inf
        previous_tolerance = math.inf
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise WorkflowError("every adaptive SCF stage must be a JSON object")
            unknown_stage = set(stage) - {
                "name", "minimum_density_rms", "maximum_iterations",
                "density_tolerance", "strict", "mixing", "fragment_mixing",
            }
            if unknown_stage:
                raise WorkflowError("adaptive SCF stage contains unsupported keys")
            _label(stage.get("name", ""), "adaptive SCF stage name")
            minimum = stage.get("minimum_density_rms")
            tolerance = stage.get("density_tolerance")
            iterations = stage.get("maximum_iterations")
            strict = stage.get("strict", False)
            if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                    or not math.isfinite(float(minimum)) or float(minimum) < 0.0
                    or float(minimum) >= previous_minimum):
                raise WorkflowError(
                    "adaptive SCF minimum_density_rms values must decrease")
            if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                    or not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0
                    or float(tolerance) > previous_tolerance):
                raise WorkflowError(
                    "adaptive SCF density tolerances must be positive and nonincreasing")
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
                raise WorkflowError(
                    "adaptive SCF maximum_iterations must be a positive integer")
            if not isinstance(strict, bool) or strict != (index == len(stages) - 1):
                raise WorkflowError("only the final adaptive SCF stage must be strict")
            stage_fragments = stage.get("fragment_mixing", {})
            if not isinstance(stage_fragments, dict) or not set(stage_fragments).issubset(labels):
                raise WorkflowError(
                    "adaptive SCF fragment_mixing contains an unknown fragment label")
            stage_settings = _mixing_settings(
                stage.get("mixing", {}), f"adaptive SCF stage {stage['name']}")
            _validate_mixing_settings(stage_settings, f"adaptive SCF stage {stage['name']}")
            for label, value in stage_fragments.items():
                fragment_settings = _mixing_settings(
                    value, f"adaptive SCF stage {stage['name']} fragment {label}")
                _validate_mixing_settings(
                    fragment_settings, f"adaptive SCF stage {stage['name']} fragment {label}")
            previous_minimum = float(minimum)
            previous_tolerance = float(tolerance)
        if float(stages[-1]["minimum_density_rms"]) != 0.0:
            raise WorkflowError("the final adaptive SCF minimum_density_rms must be zero")
        default_force = (validated_integers["maximum_freeze_thaw_cycles"]
                         - validated_integers["strict_confirmation_cycles"] + 1)
        force_strict = adaptive.get("force_strict_cycle", default_force)
        if (isinstance(force_strict, bool) or not isinstance(force_strict, int)
                or force_strict < 1 or force_strict > default_force):
            raise WorkflowError(
                "adaptive_scf.force_strict_cycle leaves too few strict confirmation cycles")
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
            embedded_scf_spin_parameters(controls, neutral[label], charge, spin)
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


def _binary_record(stream, expected: str) -> List[str]:
    raw_line = stream.readline()
    try:
        fields = raw_line.decode("ascii").split()
    except UnicodeDecodeError as error:
        raise WorkflowError("binary density metadata is not ASCII") from error
    if not fields or fields[0] != expected:
        raise WorkflowError(f"binary density expected {expected}")
    return fields[1:]


def _binary_values(stream, count: int, label: str) -> array:
    if count < 0:
        raise WorkflowError(f"binary {label} density has a negative size")
    values = array("d")
    if values.itemsize != 8:
        raise WorkflowError("binary density requires an 8-byte Python double")
    try:
        values.fromfile(stream, count)
    except EOFError as error:
        raise WorkflowError(f"binary {label} density is truncated") from error
    if sys.byteorder != "little":
        values.byteswap()
    if stream.read(1) != b"\n":
        raise WorkflowError(f"binary {label} density has an invalid terminator")
    return values


def _read_binary_density(path: Path) -> Dict[str, object]:
    try:
        with path.open("rb") as stream:
            header = _binary_record(stream, "FDE_DENSITY_BINARY")
            if len(header) != 2 or int(header[0]) != 1:
                raise WorkflowError("unsupported binary density format version")
            schema_version = int(header[1])
            fragment = _binary_record(stream, "FRAGMENT")[0]
            state = _binary_record(stream, "STATE")[0]
            geometry = _binary_record(stream, "GEOMETRY")[0]
            grid_fingerprint = _binary_record(stream, "GRID_FINGERPRINT")[0]
            pseudopotentials = _binary_record(stream, "PSEUDOPOTENTIALS")[0]
            orbitals = _binary_record(stream, "ORBITALS")[0]
            core_density = _binary_record(stream, "CORE_DENSITY")[0]
            functionals = _binary_record(stream, "FUNCTIONALS")
            grid = _binary_record(stream, "GRID")
            if len(grid) != 4:
                raise WorkflowError("binary density GRID metadata is malformed")
            grid_size = int(grid[0]) * int(grid[1]) * int(grid[2])
            populations = _binary_record(stream, "POPULATIONS")
            scf = _binary_record(stream, "SCF")
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
            energies = _binary_record(stream, "ENERGIES_RY")
            byte_order = _binary_record(stream, "BYTE_ORDER")
            if byte_order != ["LITTLE_ENDIAN"]:
                raise WorkflowError("unsupported binary density byte order")
            alpha_header = _binary_record(stream, "RHO_ALPHA_BINARY")
            alpha_size = int(alpha_header[0])
            if alpha_size != grid_size:
                raise WorkflowError("binary alpha density size does not match GRID")
            alpha = _binary_values(stream, alpha_size, "alpha")
            beta_header = _binary_record(stream, "RHO_BETA_BINARY")
            beta_size = int(beta_header[0])
            if beta_size != grid_size:
                raise WorkflowError("binary beta density size does not match GRID")
            beta = _binary_values(stream, beta_size, "beta")
            if _binary_record(stream, "END"):
                raise WorkflowError("binary density END record is malformed")
            if stream.read().strip():
                raise WorkflowError("binary density contains trailing content")
    except (IndexError, ValueError) as error:
        raise WorkflowError(f"malformed binary density artifact {path}: {error}") from error
    if grid_size <= 0 or len(alpha) != grid_size or len(beta) != grid_size:
        raise WorkflowError("binary density vector length is inconsistent")
    return {
        "fragment": fragment,
        "state": state,
        "geometry": geometry,
        "schema_version": schema_version,
        "cycle": cycle,
        "scf_converged": scf_converged,
        "scf_iterations": scf_iterations,
        "scf_density_residual": scf_density_residual,
        "initialization_seed": False,
        "artifact_format": "binary",
        "grid_size": grid_size,
        "grid_dimensions": [int(grid[0]), int(grid[1]), int(grid[2])],
        "cell_volume": float(grid[3]),
        "grid_fingerprint": grid_fingerprint,
        "pseudopotentials": pseudopotentials,
        "orbitals": orbitals,
        "core_density": core_density,
        "functionals": functionals,
        "populations": [int(populations[0]), int(populations[1])],
        "energies_ry": [float(energies[0]), float(energies[1])],
        "alpha": alpha,
        "beta": beta,
    }


def read_density(path: Path) -> Dict[str, object]:
    with path.open("rb") as stream:
        if stream.readline().split()[:1] == [b"FDE_DENSITY_BINARY"]:
            return _read_binary_density(path)
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
            populations = records["POPULATIONS"][0]
            energies = ["0", "0"]
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
            populations = records["POPULATIONS"][0]
            energies = records["ENERGIES_RY"][0]
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
            "artifact_format": "uniform" if initialization_seed else "text",
            "grid_size": len(alpha),
            "grid_dimensions": [int(grid[0]), int(grid[1]), int(grid[2])],
            "cell_volume": float(grid[3]),
            "grid_fingerprint": records["GRID_FINGERPRINT"][0][0],
            "pseudopotentials": records["PSEUDOPOTENTIALS"][0][0],
            "orbitals": records["ORBITALS"][0][0],
            "core_density": records["CORE_DENSITY"][0][0],
            "functionals": records["FUNCTIONALS"][0],
            "populations": [int(populations[0]), int(populations[1])],
            "energies_ry": [float(energies[0]), float(energies[1])],
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
    squared_difference = 0.0
    value_count = 0
    for channel in ("alpha", "beta"):
        values = first[channel]
        reference = second[channel]
        if len(values) != len(reference):
            raise WorkflowError("density residual requires equal spin-grid sizes")
        squared_difference += math.fsum(
            (left - right) ** 2 for left, right in zip(values, reference))
        value_count += len(values)
    return math.sqrt(squared_difference / value_count)


def _solve_dense_system(matrix: Sequence[Sequence[float]],
                        right_hand_side: Sequence[float]) -> List[float]:
    size = len(right_hand_side)
    work = [list(row) + [float(right_hand_side[index])]
            for index, row in enumerate(matrix)]
    if len(work) != size or any(len(row) != size + 1 for row in work):
        raise WorkflowError("outer Anderson linear system is not square")
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= 1e-18:
            raise WorkflowError("outer Anderson history is numerically singular")
        work[column], work[pivot] = work[pivot], work[column]
        inverse = 1.0 / work[column][column]
        for index in range(column, size + 1):
            work[column][index] *= inverse
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            for index in range(column, size + 1):
                work[row][index] -= factor * work[column][index]
    return [work[index][size] for index in range(size)]


def _anderson_coefficients(
        pairs: Sequence[Tuple[Mapping[str, object], Mapping[str, object]]],
        regularization: float) -> List[float]:
    for old, raw in pairs:
        density_rms(old, raw)
    count = len(pairs)
    vector_size = 2 * int(pairs[0][0]["grid_size"])
    matrix = [[0.0] * (count + 1) for _ in range(count + 1)]
    for row in range(count):
        row_old, row_raw = pairs[row]
        for column in range(row, count):
            column_old, column_raw = pairs[column]
            value = math.fsum(
                (row_output - row_input) * (column_output - column_input)
                for row_input, row_output, column_input, column_output in zip(
                    row_old["alpha"], row_raw["alpha"],
                    column_old["alpha"], column_raw["alpha"]))
            value += math.fsum(
                (row_output - row_input) * (column_output - column_input)
                for row_input, row_output, column_input, column_output in zip(
                    row_old["beta"], row_raw["beta"],
                    column_old["beta"], column_raw["beta"]))
            value /= vector_size
            matrix[row][column] = value
            matrix[column][row] = value
        matrix[row][row] += regularization
        matrix[row][count] = 1.0
        matrix[count][row] = 1.0
    solution = _solve_dense_system(matrix, [0.0] * count + [1.0])
    return solution[:count]


def _project_density(values: Iterable[float],
                     electrons: int,
                     volume_element: float) -> Tuple[array, Dict[str, object]]:
    projected = array("d")
    clipped = 0
    for value in values:
        if not math.isfinite(value):
            raise WorkflowError("outer density mixing produced a non-finite value")
        if value < 0.0:
            value = 0.0
            clipped += 1
        projected.append(value)
    integral = math.fsum(projected) * volume_element
    if electrons == 0:
        projected = array("d", [0.0]) * len(projected)
        scale = 0.0
    else:
        if not math.isfinite(integral) or integral <= 0.0:
            raise WorkflowError("outer density mixing cannot restore the spin population")
        scale = electrons / integral
        for index in range(len(projected)):
            projected[index] *= scale
    return projected, {"clipped_points": clipped, "normalization_scale": scale}


def mix_density_history(
    pairs: Sequence[Tuple[Mapping[str, object], Mapping[str, object]]],
    method: str,
    beta: float,
    regularization: float,
) -> Tuple[array, array, Dict[str, object]]:
    if not pairs:
        raise WorkflowError("outer density mixing requires at least one update")
    coefficients = ([1.0] if method == "linear"
                    else _anderson_coefficients(pairs, regularization))
    current = pairs[-1][1]
    populations = list(current["populations"])
    volume_element = float(current["cell_volume"]) / int(current["grid_size"])

    def channel_values(channel: str) -> Iterable[float]:
        for index in range(int(current["grid_size"])):
            yield math.fsum(
                coefficient
                * ((1.0 - beta) * float(old[channel][index])
                   + beta * float(raw[channel][index]))
                for coefficient, (old, raw) in zip(coefficients, pairs))

    alpha, alpha_projection = _project_density(
        channel_values("alpha"), int(populations[0]), volume_element)
    beta_density, beta_projection = _project_density(
        channel_values("beta"), int(populations[1]), volume_element)
    return alpha, beta_density, {
        "coefficients": coefficients,
        "alpha_projection": alpha_projection,
        "beta_projection": beta_projection,
    }


def _write_binary_array(stream, values: array) -> None:
    output = array("d", values)
    if sys.byteorder != "little":
        output.byteswap()
    output.tofile(stream)


def write_mixed_density(path: Path,
                        reference: Mapping[str, object],
                        alpha: array,
                        beta: array) -> None:
    dimensions = list(reference["grid_dimensions"])
    populations = list(reference["populations"])
    energies = list(reference["energies_ry"])
    functionals = list(reference["functionals"])
    iterations = max(1, int(reference.get("scf_iterations", 0)))
    residual = reference.get("scf_density_residual")
    residual = 0.0 if residual is None else float(residual)
    lines = [
        "FDE_DENSITY_BINARY 1 2",
        f"FRAGMENT {reference['fragment']}",
        f"STATE {reference['state']}",
        f"GEOMETRY {reference['geometry']}",
        f"GRID_FINGERPRINT {reference['grid_fingerprint']}",
        f"PSEUDOPOTENTIALS {reference['pseudopotentials']}",
        f"ORBITALS {reference['orbitals']}",
        f"CORE_DENSITY {reference['core_density']}",
        f"FUNCTIONALS {' '.join(str(value) for value in functionals)}",
        (f"GRID {dimensions[0]} {dimensions[1]} {dimensions[2]} "
         f"{float(reference['cell_volume']):.17g}"),
        f"POPULATIONS {populations[0]} {populations[1]}",
        (f"SCF {int(reference['cycle'])} 0 {iterations} "
         f"{residual:.17g}"),
        f"ENERGIES_RY {float(energies[0]):.17g} {float(energies[1]):.17g}",
        "BYTE_ORDER LITTLE_ENDIAN",
        f"RHO_ALPHA_BINARY {len(alpha)}",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(("\n".join(lines) + "\n").encode("ascii"))
        _write_binary_array(stream, alpha)
        stream.write(f"\nRHO_BETA_BINARY {len(beta)}\n".encode("ascii"))
        _write_binary_array(stream, beta)
        stream.write(b"\nEND\n")
    temporary.replace(path)


def apply_outer_mixing(
    controls: Mapping[str, object],
    state_directory: Path,
    cycle: int,
    strict_scf: bool,
    labels: Sequence[str],
    input_paths: Mapping[str, Path],
    raw_paths: Mapping[str, Path],
    history: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, Path], Dict[str, object]]:
    settings = controls.get("outer_mixing", {})
    method = str(settings.get("type", "none")) if isinstance(settings, dict) else "none"
    apply_in_strict = bool(settings.get("apply_in_strict", False))
    if method == "none" or (strict_scf and not apply_in_strict):
        return dict(raw_paths), {
            "type": method,
            "applied": False,
            "reason": "strict_stage" if strict_scf and method != "none" else "disabled",
            "input_densities": {label: str(input_paths[label]) for label in labels},
            "raw_densities": {label: str(raw_paths[label]) for label in labels},
            "mixed_densities": {label: str(raw_paths[label]) for label in labels},
        }

    depth = int(settings.get("history", 4))
    beta = float(settings.get("beta", 0.5))
    regularization = float(settings.get("regularization", 1e-10))
    previous_records = [entry.get("outer_mixing", {}) for entry in history]
    previous_records = [entry for entry in previous_records
                        if isinstance(entry, dict) and entry.get("applied")]
    mixed_directory = state_directory / f"cycle-{cycle:03d}" / "outer-mixed"
    mixed_directory.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}
    diagnostics: Dict[str, object] = {}
    for label in labels:
        pairs: List[Tuple[Mapping[str, object], Mapping[str, object]]] = []
        if method == "anderson" and depth > 1:
            for record in previous_records[-(depth - 1):]:
                old_path = Path(record["input_densities"][label])
                raw_path = Path(record["raw_densities"][label])
                if old_path.is_file() and raw_path.is_file():
                    pairs.append((read_density(old_path), read_density(raw_path)))
        current_old = read_density(input_paths[label])
        current_raw = read_density(raw_paths[label])
        pairs.append((current_old, current_raw))
        fallback = None
        try:
            alpha, beta_density, detail = mix_density_history(
                pairs, method, beta, regularization)
        except WorkflowError as error:
            if method != "anderson":
                raise
            alpha, beta_density, detail = mix_density_history(
                [pairs[-1]], "linear", beta, regularization)
            fallback = str(error)
        output_path = (mixed_directory / f"{label}.fde_density").resolve()
        write_mixed_density(output_path, current_raw, alpha, beta_density)
        output_paths[label] = output_path
        detail.update({"history_used": len(pairs), "fallback": fallback})
        diagnostics[label] = detail
    return output_paths, {
        "type": method,
        "applied": True,
        "beta": beta,
        "regularization": regularization,
        "input_densities": {label: str(input_paths[label]) for label in labels},
        "raw_densities": {label: str(raw_paths[label]) for label in labels},
        "mixed_densities": {label: str(output_paths[label]) for label in labels},
        "fragments": diagnostics,
    }


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
    remaining = {key.lower(): (str(value).lower() if isinstance(value, bool)
                               else str(value))
                 for key, value in values.items()}
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


def scf_schedule(controls: Mapping[str, object],
                 cycle: int,
                 previous_density_rms: float | None = None) -> Dict[str, object]:
    if cycle <= 0:
        raise WorkflowError("freeze-thaw cycle must be positive")
    adaptive = controls.get("adaptive_scf", {})
    if isinstance(adaptive, dict) and adaptive.get("enabled", False):
        stages = adaptive_scf_stages(controls)
        maximum_cycles = int(controls.get("maximum_freeze_thaw_cycles", 20))
        confirmations = int(controls.get("strict_confirmation_cycles", 2))
        force_strict = int(adaptive.get(
            "force_strict_cycle", maximum_cycles - confirmations + 1))
        if cycle >= force_strict:
            stage = stages[-1]
        elif previous_density_rms is None:
            stage = stages[0]
        else:
            stage = stages[-1]
            for candidate in stages:
                if previous_density_rms >= float(candidate["minimum_density_rms"]):
                    stage = candidate
                    break
        return {
            "mode": str(stage["name"]),
            "strict": bool(stage["strict"]),
            "maximum_iterations": int(stage["maximum_iterations"]),
            "density_tolerance": float(stage["density_tolerance"]),
            "stage": stage,
            "previous_density_rms": previous_density_rms,
        }
    inexact_cycles = int(controls.get("inexact_freeze_thaw_cycles", 0))
    strict = cycle > inexact_cycles
    if strict:
        return {
            "mode": "strict",
            "strict": True,
            "maximum_iterations": int(controls.get("maximum_scf_iterations", 100)),
            "density_tolerance": float(controls.get("scf_density_tolerance", 1e-8)),
            "stage": None,
            "previous_density_rms": previous_density_rms,
        }
    return {
        "mode": "inexact",
        "strict": False,
        "maximum_iterations": int(controls.get("inexact_scf_iterations", 50)),
        "density_tolerance": float(
            controls.get("inexact_scf_density_tolerance", 1e-3)),
        "stage": None,
        "previous_density_rms": previous_density_rms,
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


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def read_abacus_scf_metrics(job_directory: Path) -> Dict[str, object]:
    """Read machine-readable electronic-step metrics without parsing text logs."""
    candidates = sorted(job_directory.glob("OUT.*/abacus.json"))
    if not candidates:
        return {"abacus_json_available": False}
    path = candidates[-1]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        outputs = document.get("output", [])
        if not isinstance(outputs, list):
            raise ValueError("output is not an array")
        steps: List[Mapping[str, object]] = []
        for output in outputs:
            if not isinstance(output, dict):
                continue
            scf = output.get("scf", [])
            if isinstance(scf, list):
                steps.extend(step for step in scf if isinstance(step, dict))

        def finite_values(name: str) -> List[float]:
            values: List[float] = []
            for step in steps:
                value = step.get(name)
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(float(value))):
                    values.append(float(value))
            return values

        times = finite_values("time")
        residuals = finite_values("drho")
        energies = finite_values("energy")
        result: Dict[str, object] = {
            "abacus_json_available": True,
            "abacus_json": str(path.resolve()),
            "electronic_steps": len(steps),
            "electronic_step_time_seconds": math.fsum(times),
        }
        if residuals:
            result["initial_drho"] = residuals[0]
            result["final_drho"] = residuals[-1]
        if energies:
            result["initial_energy_ev"] = energies[0]
            result["final_energy_ev"] = energies[-1]
        return result
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "abacus_json_available": True,
            "abacus_json": str(path.resolve()),
            "abacus_json_error": str(error),
        }


def write_state_performance(state_directory: Path,
                            history: Sequence[Mapping[str, object]]) -> None:
    """Rewrite restart-safe JSON and JSONL performance reports for one state."""
    records: List[Dict[str, object]] = []
    for cycle in history:
        inner_scf = cycle.get("inner_scf", {})
        if not isinstance(inner_scf, dict):
            continue
        for fragment, metrics in inner_scf.items():
            if not isinstance(metrics, dict):
                continue
            record = dict(metrics)
            record.update({
                "cycle": int(cycle["cycle"]),
                "fragment": str(fragment),
                "scf_mode": str(cycle.get("scf_mode", "unknown")),
                "freeze_thaw_density_rms": cycle.get("density_rms"),
            })
            records.append(record)
    numeric_sums = {
        "wall_time_seconds": 0.0,
        "electronic_step_time_seconds": 0.0,
        "iterations": 0,
    }
    for record in records:
        for name in numeric_sums:
            value = record.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_sums[name] += value
    summary: Dict[str, object] = {
        "schema_version": 1,
        "subsystem_calls": len(records),
        "total_wall_time_seconds": numeric_sums["wall_time_seconds"],
        "total_electronic_step_time_seconds":
            numeric_sums["electronic_step_time_seconds"],
        "total_scf_iterations": int(numeric_sums["iterations"]),
        "records": records,
    }
    _atomic_json(state_directory / "performance.json", summary)
    _atomic_text(
        state_directory / "performance.jsonl",
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def write_workflow_performance(root: Path) -> None:
    """Aggregate completed state reports under a PES work directory."""
    state_reports: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/*/performance.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        state_reports.append({
            "geometry": path.parent.parent.name,
            "state": path.parent.name,
            "path": str(path.resolve()),
            "subsystem_calls": report.get("subsystem_calls", 0),
            "total_wall_time_seconds": report.get("total_wall_time_seconds", 0.0),
            "total_electronic_step_time_seconds":
                report.get("total_electronic_step_time_seconds", 0.0),
            "total_scf_iterations": report.get("total_scf_iterations", 0),
        })
    _atomic_json(root / "fde_performance.json", {
        "schema_version": 1,
        "states": state_reports,
        "total_subsystem_calls": sum(int(item["subsystem_calls"])
                                     for item in state_reports),
        "total_wall_time_seconds": math.fsum(
            float(item["total_wall_time_seconds"]) for item in state_reports),
        "total_electronic_step_time_seconds": math.fsum(
            float(item["total_electronic_step_time_seconds"])
            for item in state_reports),
        "total_scf_iterations": sum(int(item["total_scf_iterations"])
                                    for item in state_reports),
    })


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


def run_fragment_scf(
    spec: Mapping[str, object],
    geometry: Mapping[str, object],
    state: Mapping[str, object],
    state_directory: Path,
    active_label: str,
    input_densities: Mapping[str, Path],
    cycle: int,
    schedule: Mapping[str, object],
    neutral_electrons: int,
) -> Dict[str, object]:
    controls = dict(spec.get("controls", {}))
    assignment = state["fragments"][active_label]
    spin_parameters = embedded_scf_spin_parameters(
        controls,
        neutral_electrons,
        int(assignment["charge"]),
        int(assignment["spin"]))
    recovery = controls.get("mixing_recovery", {})
    recovery_enabled = bool(
        isinstance(recovery, dict) and recovery.get("enabled", False))
    maximum_retries = (len(recovery.get("fallbacks", []))
                       if recovery_enabled and isinstance(recovery, dict) else 0)
    attempts: List[Dict[str, object]] = []
    retry = 0
    attempt_densities = dict(input_densities)
    while True:
        directory_name = (active_label if retry == 0
                          else f"{active_label}-retry-{retry:02d}")
        job_directory = state_directory / f"cycle-{cycle:03d}" / directory_name
        if job_directory.exists():
            shutil.rmtree(job_directory)
        shutil.copytree(Path(geometry["template_directory"]), job_directory)
        config_path = job_directory / "FDE_CONFIG"
        write_runtime_config(config_path,
                             spec,
                             state,
                             active_label,
                             attempt_densities,
                             "result",
                             int(schedule["maximum_iterations"]),
                             float(schedule["density_tolerance"]))
        mixing = fragment_mixing_parameters(
            controls, active_label, schedule.get("stage"), retry)
        input_parameters: Dict[str, object] = {
            "calculation": "scf", "basis_type": "lcao", "gamma_only": 1,
            "nspin": spin_parameters["nspin"],
            "noncolin": 0, "lspinorb": 0, "symmetry": 0,
            "dft_functional": "pbe",
            "ks_solver": str(controls.get("ks_solver", "lapack")),
            "kpar": int(controls.get("kpar", 1)),
            "nelec": spin_parameters["nelec"],
            "nupdown": spin_parameters["nupdown"],
            "fde_task": "embedded_scf", "fde_config": "FDE_CONFIG",
            "scf_nmax": int(schedule["maximum_iterations"]),
            "scf_thr": float(schedule["density_tolerance"]),
        }
        input_parameters.update(mixing)
        patch_input(job_directory / "INPUT", input_parameters)
        log_path = job_directory / "fde_abacus.log"
        environment = dict(os.environ)
        environment.setdefault("OMP_NUM_THREADS", "1")
        launch_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                list(spec["abacus_command"]), cwd=job_directory,
                env=environment, stdout=log, stderr=subprocess.STDOUT,
                check=False)
        wall_time_seconds = time.monotonic() - launch_started
        launch_metrics: Dict[str, object] = {
            "schema_version": 1,
            "retry": retry,
            "returncode": completed.returncode,
            "wall_time_seconds": wall_time_seconds,
            "maximum_iterations": int(schedule["maximum_iterations"]),
            "density_tolerance": float(schedule["density_tolerance"]),
            "mixing": mixing,
        }
        launch_metrics.update(read_abacus_scf_metrics(job_directory))
        _atomic_json(job_directory / "fde_performance.json", launch_metrics)
        if completed.returncode != 0:
            raise WorkflowError(f"ABACUS failed in {job_directory}; see {log_path}")
        density_path, density_data = select_scf_density(
            job_directory, bool(controls.get("allow_partial_scf", False)))
        inner_converged = bool(density_data["scf_converged"])
        fragment_path = job_directory / "result.fde_fragment"
        if inner_converged and not fragment_path.is_file():
            raise WorkflowError(
                f"converged ABACUS job did not produce an FDE fragment in {job_directory}")
        if not inner_converged and fragment_path.exists():
            raise WorkflowError(
                f"partial ABACUS job produced a final FDE fragment in {job_directory}")
        if bool(controls.get("remove_abacus_restart_files", False)):
            remove_abacus_restart_files(job_directory)
        attempt_densities[active_label] = density_path.resolve()
        attempt_metrics = dict(launch_metrics)
        attempt_metrics.update({
            "converged": inner_converged,
            "density_artifact_iterations": density_data["scf_iterations"],
            "density_residual": density_data["scf_density_residual"],
            "job_directory": str(job_directory.resolve()),
        })
        attempts.append(attempt_metrics)
        if (inner_converged or not bool(schedule["strict"])
                or retry >= maximum_retries):
            break
        retry += 1

    metrics = {
        "converged": inner_converged,
        "iterations": sum(
            int(item.get("density_artifact_iterations", 0)
                or item.get("electronic_steps", 0)) for item in attempts),
        "density_residual": density_data["scf_density_residual"],
        "wall_time_seconds": math.fsum(
            float(item["wall_time_seconds"]) for item in attempts),
        "electronic_steps": sum(
            int(item.get("electronic_steps", 0)) for item in attempts),
        "electronic_step_time_seconds": math.fsum(
            float(item.get("electronic_step_time_seconds", 0.0))
            for item in attempts),
        "maximum_iterations": int(schedule["maximum_iterations"]),
        "density_tolerance": float(schedule["density_tolerance"]),
        "mixing": mixing,
        "retry_count": retry,
        "attempts": attempts,
    }
    return {
        "label": active_label,
        "density_path": density_path.resolve(),
        "density_data": density_data,
        "fragment_path": fragment_path.resolve() if inner_converged else None,
        "fragment": read_fragment(fragment_path) if inner_converged else None,
        "metrics": metrics,
    }


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
    retain_completed_cycles = int(controls.get("retain_completed_cycles", 0))
    required_strict_confirmations = int(controls.get("strict_confirmation_cycles", 2))
    update_scheme = str(controls.get("update_scheme", "auto"))
    if update_scheme == "auto":
        update_scheme = "gauss_seidel"
    jacobi_parallelism = int(controls.get("jacobi_parallelism", 1))
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
    previous_density_rms = (float(history[-1]["density_rms"]) if history else None)

    neutral = {fragment["label"]: int(fragment["neutral_valence_electrons"])
               for fragment in fragments}
    for cycle in range(start_cycle, maximum_cycles + 1):
        schedule = scf_schedule(controls, cycle, previous_density_rms)
        strict_scf = bool(schedule["strict"])
        old_density_data = {label: read_density(path) for label, path in densities.items()}
        cycle_fragments: List[Mapping[str, object]] = []
        cycle_fragment_paths: Dict[str, Path] = {}
        cycle_scf: Dict[str, Dict[str, object]] = {}
        cycle_input_paths = dict(densities)

        def launch(active_label: str,
                   input_paths: Mapping[str, Path]) -> Dict[str, object]:
            return run_fragment_scf(spec,
                                    geometry,
                                    state,
                                    state_directory,
                                    active_label,
                                    input_paths,
                                    cycle,
                                    schedule,
                                    neutral[active_label])

        results: List[Dict[str, object]] = []
        if update_scheme == "gauss_seidel":
            for active_label in update_order:
                result = launch(active_label, densities)
                results.append(result)
                densities[active_label] = Path(result["density_path"])
        elif jacobi_parallelism == 1:
            results = [launch(active_label, cycle_input_paths)
                       for active_label in update_order]
        else:
            with ThreadPoolExecutor(max_workers=jacobi_parallelism) as executor:
                futures = [executor.submit(launch, active_label, cycle_input_paths)
                           for active_label in update_order]
                results = [future.result() for future in futures]

        raw_density_paths: Dict[str, Path] = {}
        for result in results:
            active_label = str(result["label"])
            raw_density_paths[active_label] = Path(result["density_path"])
            cycle_scf[active_label] = dict(result["metrics"])
            if result["fragment_path"] is not None:
                cycle_fragment_paths[active_label] = Path(result["fragment_path"])
                cycle_fragments.append(result["fragment"])
        densities, outer_mixing = apply_outer_mixing(
            controls,
            state_directory,
            cycle,
            strict_scf,
            labels,
            cycle_input_paths,
            raw_density_paths,
            history)
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
                        "update_scheme": update_scheme,
                        "outer_mixing": outer_mixing,
                        "scf_schedule": {
                            "strict": strict_scf,
                            "maximum_iterations": schedule["maximum_iterations"],
                            "density_tolerance": schedule["density_tolerance"],
                            "previous_density_rms": schedule["previous_density_rms"],
                        },
                        "all_inner_scf_converged": all_inner_converged,
                        "strict_cycle_complete": strict_cycle_complete,
                        "strict_confirmations": strict_confirmations,
                        "inner_scf": cycle_scf})
        checkpoint: Dict[str, object] = {
            "schema_version": 1, "geometry": geometry["label"], "state": state["label"],
            "cycle": cycle, "converged": converged, "density_rms": residual,
            "energy_ry": energy, "energy_change_ry": energy_change,
            "scf_mode": schedule["mode"],
            "update_scheme": update_scheme,
            "outer_mixing": outer_mixing,
            "scf_schedule": history[-1]["scf_schedule"],
            "all_inner_scf_converged": all_inner_converged,
            "strict_cycle_complete": strict_cycle_complete,
            "strict_confirmations": strict_confirmations,
            "inner_scf": cycle_scf,
            "densities": {label: str(densities[label]) for label in labels},
            "fragments": {label: str(path) for label, path in cycle_fragment_paths.items()},
            "history": history,
        }
        _atomic_json(checkpoint_path, checkpoint)
        write_state_performance(state_directory, history)
        outer_settings = controls.get("outer_mixing", {})
        outer_history = (int(outer_settings.get("history", 4))
                         if isinstance(outer_settings, dict)
                         and outer_settings.get("type") == "anderson" else 0)
        prune_completed_cycles(
            state_directory, cycle, max(retain_completed_cycles, outer_history))
        if converged:
            composed = compose_state(str(state["label"]),
                                     [cycle_fragment_paths[label] for label in labels],
                                     float(energy), state_directory)
            checkpoint.update(composed)
            _atomic_json(checkpoint_path, checkpoint)
            return checkpoint
        previous_cycle_complete = strict_cycle_complete
        previous_complete_energy = energy if strict_cycle_complete else None
        previous_density_rms = residual
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
    write_workflow_performance(root)


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
