#!/usr/bin/env python3
"""Collect compact, provenance-rich output from an FDE Slurm array."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if not rows or not rows[0]:
        raise ValueError(f"TSV is empty: {path}")
    return rows[0], rows[1:]


def manifest_rows(
    path: Path, requested_column: str
) -> Tuple[List[str], List[Dict[str, str]], str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"manifest has no header: {path}")
    if requested_column == "auto":
        candidates = [name for name in ("spec", "spec_path") if name in fieldnames]
        if len(candidates) != 1:
            raise ValueError("manifest must contain exactly one of 'spec' or 'spec_path'")
        spec_column = candidates[0]
    else:
        spec_column = requested_column
        if spec_column not in fieldnames:
            raise ValueError(f"manifest does not contain column {spec_column!r}")
    if not rows:
        raise ValueError(f"manifest has no data rows: {path}")
    if "task_id" not in fieldnames:
        raise ValueError("manifest must contain a task_id column")
    task_ids = [row.get("task_id", "") for row in rows]
    if any(not task_id for task_id in task_ids):
        raise ValueError("manifest task_id values must be nonempty")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("manifest task_id values must be unique")
    return fieldnames, rows, spec_column


def resolve_from(path: str, parent: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = parent / candidate
    return candidate.resolve(strict=True)


def checkpoint_records(
    task_id: str,
    spec: Dict[str, Any],
    spec_path: Path,
    work_directory: Path,
) -> Tuple[List[Dict[str, Any]], bool]:
    records: List[Dict[str, Any]] = []
    all_converged = True
    for geometry in spec.get("geometries", []):
        geometry_label = str(geometry["label"])
        for state in spec.get("states", []):
            state_label = str(state["label"])
            state_directory = work_directory / geometry_label / state_label
            checkpoint_path = state_directory / "checkpoint.json"
            if not checkpoint_path.is_file():
                all_converged = False
                records.append(
                    {
                        "task_id": task_id,
                        "geometry": geometry_label,
                        "state": state_label,
                        "converged": False,
                        "checkpoint": "",
                        "spec": str(spec_path),
                    }
                )
                continue
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            converged = checkpoint.get("converged") is True
            all_converged = all_converged and converged
            record: Dict[str, Any] = {
                "task_id": task_id,
                "geometry": geometry_label,
                "state": state_label,
                "converged": converged,
                "cycle": checkpoint.get("cycle", ""),
                "density_rms": checkpoint.get("density_rms", ""),
                "energy_change_ry": checkpoint.get("energy_change_ry", ""),
                "energy_ry": checkpoint.get("energy_ry", ""),
                "total_scf_iterations": "",
                "total_wall_time_seconds": "",
                "checkpoint": str(checkpoint_path),
                "spec": str(spec_path),
            }
            performance_path = state_directory / "performance.json"
            if performance_path.is_file():
                performance = json.loads(performance_path.read_text(encoding="utf-8"))
                record["total_scf_iterations"] = performance.get(
                    "total_scf_iterations", ""
                )
                record["total_wall_time_seconds"] = performance.get(
                    "total_wall_time_seconds", ""
                )
            records.append(record)
    return records, all_converged


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def collect(
    manifest: Path,
    output: Path,
    *,
    spec_column: str = "auto",
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    manifest = manifest.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, rows, selected_column = manifest_rows(manifest, spec_column)

    aggregate_header: List[str] | None = None
    aggregate_rows: List[List[str]] = []
    point_records: List[Dict[str, Any]] = []
    convergence: List[Dict[str, Any]] = []
    specs_to_copy: List[Tuple[str, Path]] = []
    incomplete = 0

    for index, row in enumerate(rows):
        task_id = row["task_id"]
        raw_spec = row.get(selected_column, "")
        if not raw_spec:
            raise ValueError(f"manifest row {index + 2} has an empty spec path")
        spec_path = resolve_from(raw_spec, manifest.parent)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        raw_work = str(spec.get("work_directory", ""))
        if not raw_work:
            raise ValueError(f"workflow has no work_directory: {spec_path}")
        work_directory = resolve_from(raw_work, spec_path.parent)
        state_records, all_converged = checkpoint_records(
            task_id, spec, spec_path, work_directory
        )
        if not state_records:
            raise ValueError(f"workflow has no geometry/state combinations: {spec_path}")
        convergence.extend(state_records)
        specs_to_copy.append((task_id, spec_path))
        pes_path = work_directory / "fde_pes.tsv"
        complete = all_converged and pes_path.is_file()
        if not complete:
            incomplete += 1
            point_records.append(
                {
                    "task_id": task_id,
                    "status": "incomplete",
                    "spec": str(spec_path),
                    "work_directory": str(work_directory),
                    "spec_sha256": sha256(spec_path),
                    "pes_sha256": "",
                }
            )
            if not allow_incomplete:
                raise RuntimeError(
                    f"task {task_id} is incomplete or lacks fde_pes.tsv: {work_directory}"
                )
            continue

        header, pes_rows = read_tsv(pes_path)
        if aggregate_header is None:
            aggregate_header = header
        elif header != aggregate_header:
            raise ValueError(f"PES header differs in {pes_path}")
        aggregate_rows.extend(pes_rows)
        point_records.append(
            {
                "task_id": task_id,
                "status": "complete",
                "spec": str(spec_path),
                "work_directory": str(work_directory),
                "spec_sha256": sha256(spec_path),
                "pes_sha256": sha256(pes_path),
            }
        )

    if aggregate_header is None:
        raise RuntimeError("no complete fde_pes.tsv files were found")

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        write_tsv(stage / "fde_pes.tsv", aggregate_header, aggregate_rows)
        point_header = (
            "task_id",
            "status",
            "spec",
            "work_directory",
            "spec_sha256",
            "pes_sha256",
        )
        write_tsv(
            stage / "points.tsv",
            point_header,
            ([record[name] for name in point_header] for record in point_records),
        )
        convergence_header = (
            "task_id",
            "geometry",
            "state",
            "converged",
            "cycle",
            "density_rms",
            "energy_change_ry",
            "energy_ry",
            "total_scf_iterations",
            "total_wall_time_seconds",
            "checkpoint",
            "spec",
        )
        write_tsv(
            stage / "convergence.tsv",
            convergence_header,
            (
                [record.get(name, "") for name in convergence_header]
                for record in convergence
            ),
        )
        shutil.copy2(manifest, stage / "source_manifest.tsv")
        copied_specs = stage / "specs"
        copied_specs.mkdir()
        for task_id, spec_path in specs_to_copy:
            safe_task = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in task_id
            )
            shutil.copy2(spec_path, copied_specs / f"{safe_task}-{spec_path.name}")

        summary = {
            "schema_version": 1,
            "source_manifest": str(manifest),
            "source_manifest_sha256": sha256(manifest),
            "manifest_columns": fieldnames,
            "spec_column": selected_column,
            "tasks": len(rows),
            "complete_tasks": len(rows) - incomplete,
            "incomplete_tasks": incomplete,
            "pes_rows": len(aggregate_rows),
            "fde_pes_sha256": sha256(stage / "fde_pes.tsv"),
            "convergence_sha256": sha256(stage / "convergence.tsv"),
            "points_sha256": sha256(stage / "points.tsv"),
        }
        (stage / "collection.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    return summary


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spec-column", default="auto")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    summary = collect(
        args.manifest,
        args.output,
        spec_column=args.spec_column,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
