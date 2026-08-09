#!/usr/bin/env python3
"""Compare primitive-cell k bands with a folded supercell spectrum."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence


class BandFoldingError(RuntimeError):
    """Raised when a band artifact or folding contract is invalid."""


def _record(lines: Sequence[str], index: int, token: str) -> List[str]:
    if index >= len(lines):
        raise BandFoldingError(f"missing {token} record")
    fields = lines[index].split()
    if not fields or fields[0] != token:
        found = fields[0] if fields else "an empty record"
        raise BandFoldingError(f"expected {token}, found {found}")
    return fields[1:]


def read_band_artifact(path: Path) -> Dict[str, object]:
    """Read and validate the text FDE_KPOINT_BANDS schema."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    try:
        header = _record(lines, 0, "FDE_KPOINT_BANDS")
        if header != ["1"]:
            raise BandFoldingError("unsupported FDE k-point band schema")
        state = _record(lines, 1, "STATE")[0]
        fragment = _record(lines, 2, "FRAGMENT")[0]
        geometry = _record(lines, 3, "GEOMETRY")[0]
        orbitals = _record(lines, 4, "ORBITALS")[0]
        ao_dimension = int(_record(lines, 5, "AO_DIMENSION")[0])
        band_count = int(_record(lines, 6, "BAND_COUNT")[0])
        point_count = int(_record(lines, 7, "KPOINT_COUNT")[0])
    except (IndexError, ValueError) as error:
        raise BandFoldingError(f"malformed header in {path}: {error}") from error
    if (not state or not fragment or not geometry or not orbitals
            or ao_dimension <= 0 or band_count <= 0
            or band_count > ao_dimension or point_count <= 0
            or point_count % 2):
        raise BandFoldingError(f"invalid k-point band header in {path}")

    points: List[Dict[str, object]] = []
    line_index = 8
    for _ in range(point_count):
        try:
            kpoint = _record(lines, line_index, "KPOINT")
            eigenvalues = _record(lines, line_index + 1, "EIGENVALUES")
            spin = int(kpoint[0])
            physical_index = int(kpoint[1])
            coordinates = [float(value) for value in kpoint[2:5]]
            weight = float(kpoint[5])
            count = int(eigenvalues[0])
            energies = [float(value) for value in eigenvalues[1:]]
        except (IndexError, ValueError) as error:
            raise BandFoldingError(
                f"malformed k-point record in {path}: {error}") from error
        if (spin not in (0, 1) or physical_index < 0 or len(kpoint) != 6
                or count != band_count or len(energies) != count
                or not all(math.isfinite(value)
                           for value in coordinates + [weight] + energies)
                or weight < 0.0
                or any(energies[index] < energies[index - 1] - 1.0e-12
                       for index in range(1, len(energies)))):
            raise BandFoldingError(f"invalid k-point record in {path}")
        points.append({
            "spin": spin,
            "index": physical_index,
            "k": coordinates,
            "weight": weight,
            "eigenvalues_ry": energies,
        })
        line_index += 2
    if _record(lines, line_index, "END_FDE_KPOINT_BANDS"):
        raise BandFoldingError(f"END_FDE_KPOINT_BANDS has fields in {path}")
    if line_index + 1 != len(lines):
        raise BandFoldingError(f"trailing content in {path}")

    by_spin = {spin: [point for point in points if point["spin"] == spin]
               for spin in (0, 1)}
    if not by_spin[0] or len(by_spin[0]) != len(by_spin[1]):
        raise BandFoldingError(f"unpaired spin meshes in {path}")
    for spin in (0, 1):
        if [point["index"] for point in by_spin[spin]] != list(range(len(by_spin[spin]))):
            raise BandFoldingError(f"noncontiguous k-point indices in {path}")
        if abs(math.fsum(point["weight"] for point in by_spin[spin]) - 1.0) > 1.0e-10:
            raise BandFoldingError(f"unnormalized k-point weights in {path}")
    for alpha, beta in zip(by_spin[0], by_spin[1]):
        if (max(abs(left - right) for left, right in zip(alpha["k"], beta["k"])) > 1.0e-10
                or abs(alpha["weight"] - beta["weight"]) > 1.0e-10):
            raise BandFoldingError(f"alpha/beta k meshes differ in {path}")
    return {
        "state": state,
        "fragment": fragment,
        "geometry": geometry,
        "orbitals": orbitals,
        "ao_dimension": ao_dimension,
        "band_count": band_count,
        "by_spin": by_spin,
    }


def compare_band_folding(primitive: Mapping[str, object],
                         supercell: Mapping[str, object],
                         tolerance_ry: float) -> Dict[str, object]:
    """Return a deterministic spin-resolved sorted-spectrum comparison."""
    if not math.isfinite(tolerance_ry) or tolerance_ry <= 0.0:
        raise BandFoldingError("tolerance_ry must be finite and positive")
    for field in ("state", "fragment", "orbitals"):
        if primitive[field] != supercell[field]:
            raise BandFoldingError(f"primitive and supercell {field} differ")
    primitive_mesh = primitive["by_spin"]
    supercell_mesh = supercell["by_spin"]
    primitive_kpoints = len(primitive_mesh[0])
    supercell_kpoints = len(supercell_mesh[0])
    if primitive_kpoints <= supercell_kpoints:
        raise BandFoldingError(
            "primitive mesh must contain more k points than the supercell mesh")
    if any(max(abs(value) for value in point["k"]) > 1.0e-10
           for spin in (0, 1) for point in supercell_mesh[spin]):
        raise BandFoldingError("supercell acceptance spectrum must be at Gamma")
    primitive_size = primitive_kpoints * int(primitive["band_count"])
    supercell_size = supercell_kpoints * int(supercell["band_count"])
    if primitive_size != supercell_size:
        raise BandFoldingError(
            "primitive and supercell folded spectra have different sizes")

    spin_results = []
    all_differences: List[float] = []
    for spin in (0, 1):
        primitive_values = sorted(
            value for point in primitive_mesh[spin]
            for value in point["eigenvalues_ry"])
        supercell_values = sorted(
            value for point in supercell_mesh[spin]
            for value in point["eigenvalues_ry"])
        differences = [abs(left - right)
                       for left, right in zip(primitive_values, supercell_values)]
        all_differences.extend(differences)
        spin_results.append({
            "spin": spin,
            "eigenvalue_count": len(differences),
            "maximum_absolute_error_ry": max(differences),
            "rms_error_ry": math.sqrt(
                math.fsum(value * value for value in differences) / len(differences)),
        })
    maximum_error = max(all_differences)
    return {
        "schema_version": 1,
        "passed": maximum_error <= tolerance_ry,
        "tolerance_ry": tolerance_ry,
        "maximum_absolute_error_ry": maximum_error,
        "primitive_kpoints_per_spin": primitive_kpoints,
        "supercell_kpoints_per_spin": supercell_kpoints,
        "folding_factor": primitive_kpoints // supercell_kpoints,
        "spins": spin_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive", type=Path, required=True)
    parser.add_argument("--supercell", type=Path, required=True)
    parser.add_argument("--tolerance-ry", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = compare_band_folding(
            read_band_artifact(arguments.primitive),
            read_band_artifact(arguments.supercell),
            arguments.tolerance_ry)
    except (BandFoldingError, OSError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
