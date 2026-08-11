#!/usr/bin/env python3
"""Prepare the single-point UKS FDE example and its compact seed densities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Sequence, Tuple


ANGSTROM_TO_BOHR = 1.8897261254578281
ROOT = Path(__file__).resolve().parent
DEFAULT_RESOURCE_ROOT = ROOT / "resources"


def determinant(rows: Sequence[Sequence[float]]) -> float:
    return (rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]))


def cube_grid(path: Path) -> Tuple[int, int, int, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError("cube file does not contain a complete grid header")
    axes: List[List[float]] = []
    counts: List[int] = []
    for line in lines[3:6]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("cube grid-vector record is malformed")
        signed_count = int(fields[0])
        scale = ANGSTROM_TO_BOHR if signed_count < 0 else 1.0
        count = abs(signed_count)
        if count == 0:
            raise ValueError("cube grid dimension is zero")
        counts.append(count)
        axes.append([float(fields[index]) * count * scale for index in range(1, 4)])
    volume = abs(determinant(axes))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("cube cell volume is invalid")
    return counts[0], counts[1], counts[2], volume


def populations(neutral: int, charge: int, spin: int) -> Tuple[int, int]:
    electrons = neutral - charge
    if electrons < 0 or abs(spin) > electrons or (electrons + spin) % 2:
        raise ValueError("invalid charge/spin population")
    return (electrons + spin) // 2, (electrons - spin) // 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def load_resource_manifest() -> Dict[str, object]:
    return json.loads((ROOT / "default_resources.json").read_text(encoding="utf-8"))


def validate_resources(pseudo_dir: Path, orbital_dir: Path,
                       manifest: Dict[str, object], verify: bool) -> None:
    for resource in manifest["resources"]:
        directory = pseudo_dir if resource["kind"] == "pseudopotentials" else orbital_dir
        path = directory / Path(resource["path"]).name
        if not path.is_file():
            raise FileNotFoundError(f"required {resource['kind']} file is missing: {path}")
        if verify:
            actual = sha256(path)
            if actual != resource["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {path}: expected {resource['sha256']}, got {actual}")


def patch_input_text(text: str, values: Dict[str, object]) -> str:
    remaining = {key.lower(): str(value) for key, value in values.items()}
    output: List[str] = []
    for line in text.splitlines():
        fields = line.strip().split()
        key = fields[0].lower() if fields and not fields[0].startswith("#") else ""
        if key in remaining:
            leading = line[:len(line) - len(line.lstrip())]
            output.append(f"{leading}{key:<24} {remaining.pop(key)}")
        else:
            output.append(line)
    for key in sorted(remaining):
        output.append(f"{key:<24} {remaining[key]}")
    return "\n".join(output) + "\n"


def probe_grid(root: Path, abacus: Path, pseudo_dir: Path,
               orbital_dir: Path,
               input_overrides: Dict[str, object]) -> Tuple[int, int, int, float]:
    log_path = root / "grid_probe.log"
    with tempfile.TemporaryDirectory(prefix="fde-grid-probe-", dir=str(root)) as directory:
        work = Path(directory)
        shutil.copy2(root / "STRU", work / "STRU")
        shutil.copy2(root / "KPT", work / "KPT")
        input_text = (root / "INPUT").read_text(encoding="utf-8")
        probe_overrides = dict(input_overrides)
        probe_overrides.update({
            "suffix": "fde_grid_probe",
            "scf_nmax": 1,
            "nelec": 22,
            "nspin": 1,
            "nbands": 12,
            "out_chg": "2 10",
            "fde_task": "none",
            "pseudo_dir": pseudo_dir,
            "orbital_dir": orbital_dir,
        })
        (work / "INPUT").write_text(
            patch_input_text(input_text, probe_overrides), encoding="utf-8"
        )
        environment = dict(os.environ)
        environment.setdefault("OMP_NUM_THREADS", "1")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run([str(abacus)], cwd=work, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        candidates = []
        for name in ("chg_ini.cube", "chgs1_ini.cube", "SPIN1_CHG_INI.cube",
                     "chg.cube", "chgs1.cube", "SPIN1_CHG.cube"):
            candidates.extend(work.glob(f"OUT.*/{name}"))
        if not candidates:
            raise RuntimeError(
                f"ABACUS grid probe returned {completed.returncode} without a cube; see {log_path}")
        if completed.returncode != 0:
            raise RuntimeError(
                f"ABACUS grid probe returned {completed.returncode}; see {log_path}")
        return cube_grid(candidates[0])


def write_seed(path: Path,
               geometry: str,
               state: str,
               fragment: str,
               alpha: int,
               beta: int,
               grid: Tuple[int, int, int, float],
               resource_commit: str,
               kedf: str) -> None:
    nx, ny, nz, volume = grid
    lines = ["FDE_UNIFORM_DENSITY_SEED 1", f"FRAGMENT {fragment}", f"STATE {state}",
             f"GEOMETRY {geometry}", f"GRID_FINGERPRINT grid-{nx}x{ny}x{nz}",
             f"PSEUDOPOTENTIALS sg15-v1.0-pbe-oncv-no-nlcc-{resource_commit}",
             f"ORBITALS standard-v2.0-dzp-100ry-{resource_commit}",
             "CORE_DENSITY none", f"FUNCTIONALS pbe {kedf}",
             f"GRID {nx} {ny} {nz} {volume:.17g}", f"POPULATIONS {alpha} {beta}",
             f"RHO_UNIFORM {alpha / volume:.17g} {beta / volume:.17g}", "END"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_stru(template: str, f_c: float, c_cl: float) -> str:
    f_line = "9.80263158 12.000000 12.000000 0 0 0"
    cl_line = "14.12368421 12.000000 12.000000 0 0 0"
    if f_line not in template or cl_line not in template:
        raise ValueError("template STRU coordinate anchors are missing")
    result = template.replace(f_line,
                              f"{12.0 - f_c:.8f} 12.000000 12.000000 0 0 0")
    return result.replace(cl_line,
                          f"{12.0 + c_cl:.8f} 12.000000 12.000000 0 0 0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument(
        "--mpi-ranks", type=positive_integer, default=1,
        help="run each embedded SCF with mpirun at this rank count (default: 1)")
    parser.add_argument("--pseudo-dir", type=Path,
                        default=DEFAULT_RESOURCE_ROOT / "pseudopotentials")
    parser.add_argument("--orbital-dir", type=Path,
                        default=DEFAULT_RESOURCE_ROOT / "orbitals")
    parser.add_argument(
        "--grid-cube", type=Path,
        help="existing matching cube; omit to run a one-iteration ABACUS grid probe")
    parser.add_argument(
        "--allow-unverified-resources", action="store_true",
        help="allow custom files with the default names without SHA-256 verification")
    parser.add_argument(
        "--kedf", choices=("thomas_fermi", "pw91k", "revapbek"),
        default="pw91k",
        help="nonadditive kinetic functional (default: pw91k)")
    parser.add_argument(
        "--variant-label",
        help="safe suffix for independent benchmark workflow/generated/work paths")
    parser.add_argument("--ecutwfc", type=positive_float, default=40.0)
    parser.add_argument(
        "--scf-density-tolerance", type=positive_float, default=3e-6)
    parser.add_argument(
        "--freeze-thaw-density-tolerance", type=positive_float, default=1e-5)
    parser.add_argument(
        "--energy-tolerance-ry", type=positive_float, default=2e-5)
    parser.add_argument(
        "--maximum-scf-iterations", type=positive_integer, default=200)
    arguments = parser.parse_args()

    if (arguments.variant_label is not None
            and re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
                         arguments.variant_label) is None):
        parser.error("--variant-label contains unsafe characters")

    root = ROOT
    abacus = arguments.abacus.resolve()
    pseudo_dir = arguments.pseudo_dir.resolve()
    orbital_dir = arguments.orbital_dir.resolve()
    for path, description in ((abacus, "ABACUS executable"),
                              (pseudo_dir, "pseudopotential directory"),
                              (orbital_dir, "orbital directory")):
        if not path.exists():
            parser.error(f"{description} does not exist: {path}")
        if any(character.isspace() for character in str(path)):
            parser.error(f"{description} path must not contain whitespace: {path}")
    if not abacus.is_file():
        parser.error(f"ABACUS executable is not a file: {abacus}")
    if not os.access(abacus, os.X_OK):
        parser.error(f"ABACUS executable is not executable: {abacus}")
    launcher = None
    if arguments.mpi_ranks > 1:
        launcher = shutil.which("mpirun")
        if launcher is None:
            parser.error("mpirun is required when --mpi-ranks is greater than one")
    manifest = load_resource_manifest()
    try:
        validate_resources(pseudo_dir, orbital_dir, manifest,
                           not arguments.allow_unverified_resources)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        if arguments.grid_cube:
            grid_cube = arguments.grid_cube.resolve()
            if not grid_cube.is_file():
                parser.error(f"grid cube does not exist: {grid_cube}")
            if any(character.isspace() for character in str(grid_cube)):
                parser.error(f"grid cube path must not contain whitespace: {grid_cube}")
            grid = cube_grid(grid_cube)
        else:
            grid = probe_grid(
                root,
                abacus,
                pseudo_dir,
                orbital_dir,
                {
                    "ecutwfc": arguments.ecutwfc,
                    "scf_thr": arguments.scf_density_tolerance,
                    "scf_nmax": arguments.maximum_scf_iterations,
                },
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    spec = json.loads((root / "workflow.template.json").read_text(encoding="utf-8"))
    spec["abacus_command"] = ([launcher, "-np", str(arguments.mpi_ranks), str(abacus)]
                               if launcher else [str(abacus)])
    spec["postprocess_command"] = [str(abacus)]
    spec["controls"]["kedf"] = arguments.kedf
    spec["controls"]["maximum_scf_iterations"] = arguments.maximum_scf_iterations
    spec["controls"]["scf_density_tolerance"] = arguments.scf_density_tolerance
    spec["controls"]["freeze_thaw_density_tolerance"] \
        = arguments.freeze_thaw_density_tolerance
    spec["controls"]["energy_tolerance_ry"] = arguments.energy_tolerance_ry
    adaptive_stages = spec["controls"].get("adaptive_scf", {}).get("stages", [])
    for stage in adaptive_stages:
        if stage.get("strict"):
            stage["maximum_iterations"] = arguments.maximum_scf_iterations
            stage["density_tolerance"] = arguments.scf_density_tolerance
    if arguments.variant_label is not None:
        variant_suffix = f"-{arguments.variant_label}"
    else:
        variant_suffix = "" if arguments.kedf == "pw91k" else f"-{arguments.kedf}"
    spec["work_directory"] = str((root / f"work{variant_suffix}").resolve())
    spec.setdefault("provenance", {})["numerical_controls"] = {
        "ecutwfc_ry": arguments.ecutwfc,
        "scf_density_tolerance": arguments.scf_density_tolerance,
        "freeze_thaw_density_tolerance": arguments.freeze_thaw_density_tolerance,
        "energy_tolerance_ry": arguments.energy_tolerance_ry,
        "maximum_scf_iterations": arguments.maximum_scf_iterations,
    }
    spec["geometries"] = []
    generated = root / f"generated{variant_suffix}"
    generated.mkdir(exist_ok=True)
    stru_template = (root / "STRU").read_text(encoding="utf-8")

    with (root / "geometry.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label = row["label"]
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", label) is None:
                raise ValueError(f"unsafe geometry label: {label}")
            geometry_root = generated / label
            template_output = geometry_root / "template"
            if template_output.exists():
                shutil.rmtree(template_output)
            template_output.mkdir(parents=True)
            for filename in ("INPUT", "KPT", "STRU"):
                shutil.copy2(root / filename, template_output / filename)
            input_path = template_output / "INPUT"
            input_path.write_text(
                patch_input_text(
                    input_path.read_text(encoding="utf-8"),
                    {
                        "ecutwfc": arguments.ecutwfc,
                        "scf_thr": arguments.scf_density_tolerance,
                        "scf_nmax": arguments.maximum_scf_iterations,
                    },
                ),
                encoding="utf-8",
            )
            (template_output / "STRU").write_text(
                render_stru(stru_template, float(row["f_c_angstrom"]),
                            float(row["c_cl_angstrom"])), encoding="utf-8")
            with (template_output / "INPUT").open("a", encoding="utf-8") as input_file:
                input_file.write(f"pseudo_dir               {pseudo_dir}\n")
                input_file.write(f"orbital_dir              {orbital_dir}\n")

            seed_directory = geometry_root / "seeds"
            seed_directory.mkdir(parents=True, exist_ok=True)
            initial: Dict[str, Dict[str, str]] = {}
            for state in spec["states"]:
                state_label = state["label"]
                initial[state_label] = {}
                for fragment in spec["fragments"]:
                    fragment_label = fragment["label"]
                    assignment = state["fragments"][fragment_label]
                    alpha, beta = populations(fragment["neutral_valence_electrons"],
                                              assignment["charge"], assignment["spin"])
                    seed_path = seed_directory / f"{state_label}_{fragment_label}.fde_seed"
                    write_seed(seed_path, label, state_label, fragment_label,
                               alpha, beta, grid, manifest["source_commit"],
                               arguments.kedf)
                    initial[state_label][fragment_label] = str(seed_path.resolve())
            spec["geometries"].append({"label": label,
                                       "coordinate_angstrom": (float(row["c_cl_angstrom"])
                                                               - float(row["f_c_angstrom"])),
                                       "template_directory": str(template_output.resolve()),
                                       "initial_densities": initial})

    workflow_path = root / f"workflow{variant_suffix}.json"
    workflow_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(spec['geometries'])} geometries with grid {grid[:3]}.")
    print(f"Workflow: {workflow_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
