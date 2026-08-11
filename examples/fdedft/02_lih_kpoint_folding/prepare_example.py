#!/usr/bin/env python3
"""Generate absolute FDE density paths for the LiH folding acceptance case."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent


SYSTEMS = {
    "primitive": {
        "atom_count": 2,
        "li_atoms": "1 0",
        "h_atoms": "1 1",
        "li_electrons": 1,
        "h_electrons": 1,
        "grid": (48, 48, 48),
        "volume": 431.8934082249199,
        "geometry": "lih-primitive",
    },
    "supercell": {
        "atom_count": 4,
        "li_atoms": "2 0 1",
        "h_atoms": "2 2 3",
        "li_electrons": 2,
        "h_electrons": 2,
        "grid": (96, 48, 48),
        "volume": 863.7868164498398,
        "geometry": "lih-supercell",
    },
}


def write_seed(path: Path, fragment: str, geometry: str,
               grid: Sequence[int], volume: float,
               alpha: int, beta: int) -> None:
    density_alpha = alpha / volume
    density_beta = beta / volume
    path.write_text(
        "\n".join((
            "FDE_UNIFORM_DENSITY_SEED 1",
            f"FRAGMENT {fragment}",
            "STATE fold",
            f"GEOMETRY {geometry}",
            f"GRID_FINGERPRINT grid-{grid[0]}x{grid[1]}x{grid[2]}",
            "PSEUDOPOTENTIALS lih-sg15-pbe",
            "ORBITALS lih-dzp",
            "CORE_DENSITY none",
            "FUNCTIONALS pbe pw91k",
            f"GRID {grid[0]} {grid[1]} {grid[2]} {volume:.17g}",
            f"POPULATIONS {alpha} {beta}",
            f"RHO_UNIFORM {density_alpha:.17g} {density_beta:.17g}",
            "END",
        )) + "\n",
        encoding="utf-8",
    )


def prepare(root: Path = ROOT) -> None:
    for required in (
        root / "resources/pseudopotentials/H_ONCV_PBE-1.0.upf",
        root / "resources/pseudopotentials/Li_ONCV_PBE-1.0.upf",
        root / "resources/orbitals/H_gga_8au_100Ry_2s1p.orb",
        root / "resources/orbitals/Li_gga_8au_100Ry_4s1p.orb",
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"missing pinned resource {required}; run fetch_default_resources.py")

    for name, values in SYSTEMS.items():
        directory = (root / name).resolve(strict=True)
        li_seed = directory / "li.fde_seed"
        h_seed = directory / "h.fde_seed"
        li_electrons = int(values["li_electrons"])
        h_electrons = int(values["h_electrons"])
        grid = values["grid"]
        volume = float(values["volume"])
        geometry = str(values["geometry"])
        write_seed(
            li_seed, "Li", geometry, grid, volume, li_electrons, 0)
        write_seed(
            h_seed, "H", geometry, grid, volume, 0, h_electrons)
        (directory / "FDE_CONFIG").write_text(
            "\n".join((
                "FDE_CONFIG 1",
                f"ATOM_COUNT {values['atom_count']}",
                f"FRAGMENT Li {li_electrons} {values['li_atoms']}",
                f"FRAGMENT H {h_electrons} {values['h_atoms']}",
                f"STATE fold 0 0 2 Li 0 {li_electrons} H 0 {-h_electrons}",
                f"STATE reverse 0 0 2 Li 0 {-li_electrons} H 0 {h_electrons}",
                "ACTIVE_STATE fold",
                "ACTIVE_FRAGMENT Li",
                f"ACTIVE_DENSITY {li_seed}",
                f"FROZEN_DENSITY H {h_seed}",
                "OUTPUT_PREFIX result",
                "FRAGMENT_XC pbe",
                "EMBEDDING_XC pbe",
                "COUPLING_PROVIDER symmetric_linearized",
                "TRANSITION_DENSITY_TRACE_TOLERANCE 1e-8",
                "KEDF pw91k",
                "DENSITY_FLOOR_BOHR3 1e-12",
                "MAX_SCF_ITERATIONS 240",
                "SCF_DENSITY_TOLERANCE 1e-8",
                "ELECTRON_TOLERANCE 1e-8",
                "MIXING_BETA 0.10",
                "MAX_FREEZE_THAW_CYCLES 2",
                "FREEZE_THAW_DENSITY_TOLERANCE 1e-8",
                "ENERGY_TOLERANCE_RY 1e-10",
                "UPDATE_ORDER 2 Li H",
                "K_STATES 2 fold reverse",
                "L_FRAGMENTS 2 Li H",
                "M_FRAGMENTS 2 Li H",
                "SINGULAR_VALUE_TOLERANCE 1e-10",
                "OVERLAP_EIGENVALUE_CUTOFF 1e-9",
                "SYMMETRY_TOLERANCE 1e-10",
                "RESIDUAL_TOLERANCE 1e-8",
                "MINIMUM_ROOT_OVERLAP 0.5",
                "CALCULATE_FORCE false",
                "HARTREE_RECIPROCITY_TOLERANCE_RY 1e-8",
                "END_FDE_CONFIG",
            )) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        prepare()
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print("Prepared primitive and 2 x 1 x 1 supercell FDE inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
