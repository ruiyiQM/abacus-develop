#!/usr/bin/env python3
"""Prepare absolute paths and compact FDE seed densities for this example."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
from typing import Dict, List, Sequence, Tuple


ANGSTROM_TO_BOHR = 1.8897261254578281


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


def write_seed(path: Path,
               geometry: str,
               state: str,
               fragment: str,
               alpha: int,
               beta: int,
               grid: Tuple[int, int, int, float]) -> None:
    nx, ny, nz, volume = grid
    lines = ["FDE_UNIFORM_DENSITY_SEED 1", f"FRAGMENT {fragment}", f"STATE {state}",
             f"GEOMETRY {geometry}", f"GRID_FINGERPRINT grid-{nx}x{ny}x{nz}",
             "PSEUDOPOTENTIALS pbe-oncv-no-nlcc", "ORBITALS nao-100ry",
             "CORE_DENSITY none", "FUNCTIONALS pbe lc94",
             f"GRID {nx} {ny} {nz} {volume:.17g}", f"POPULATIONS {alpha} {beta}",
             f"RHO_UNIFORM {alpha / volume:.17g} {beta / volume:.17g}", "END"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_stru(template: str, f_c: float, c_cl: float) -> str:
    f_line = "9.850000 12.000000 12.000000 0 0 0"
    cl_line = "14.150000 12.000000 12.000000 0 0 0"
    if f_line not in template or cl_line not in template:
        raise ValueError("template STRU coordinate anchors are missing")
    result = template.replace(f_line,
                              f"{12.0 - f_c:.6f} 12.000000 12.000000 0 0 0")
    return result.replace(cl_line,
                          f"{12.0 + c_cl:.6f} 12.000000 12.000000 0 0 0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--orbital-dir", type=Path, required=True)
    parser.add_argument("--grid-cube", type=Path, required=True)
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parent
    abacus = arguments.abacus.resolve()
    pseudo_dir = arguments.pseudo_dir.resolve()
    orbital_dir = arguments.orbital_dir.resolve()
    grid_cube = arguments.grid_cube.resolve()
    for path, description in ((abacus, "ABACUS executable"),
                              (pseudo_dir, "pseudopotential directory"),
                              (orbital_dir, "orbital directory"),
                              (grid_cube, "grid cube")):
        if not path.exists():
            parser.error(f"{description} does not exist: {path}")
        if any(character.isspace() for character in str(path)):
            parser.error(f"{description} path must not contain whitespace: {path}")
    grid = cube_grid(grid_cube)

    spec = json.loads((root / "workflow.template.json").read_text(encoding="utf-8"))
    spec["abacus_command"] = [str(abacus)]
    spec["postprocess_command"] = [str(abacus)]
    spec["work_directory"] = str((root / "work").resolve())
    spec["geometries"] = []
    generated = root / "generated"
    generated.mkdir(exist_ok=True)
    template = root / "template"
    stru_template = (template / "STRU").read_text(encoding="utf-8")

    with (root / "geometries.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label = row["label"]
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", label) is None:
                raise ValueError(f"unsafe geometry label: {label}")
            geometry_root = generated / label
            template_output = geometry_root / "template"
            if template_output.exists():
                shutil.rmtree(template_output)
            shutil.copytree(template, template_output)
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
                               alpha, beta, grid)
                    initial[state_label][fragment_label] = str(seed_path.resolve())
            spec["geometries"].append({"label": label,
                                       "coordinate_angstrom": (float(row["c_cl_angstrom"])
                                                               - float(row["f_c_angstrom"])),
                                       "template_directory": str(template_output.resolve()),
                                       "initial_densities": initial})

    (root / "workflow.json").write_text(json.dumps(spec, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"Prepared {len(spec['geometries'])} geometries with grid {grid[:3]}.")
    print(f"Workflow: {root / 'workflow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
