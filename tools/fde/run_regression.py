#!/usr/bin/env python3
"""Run the durable offline and optional native ABACUS FDE regression gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Mapping, Sequence


class RegressionError(RuntimeError):
    """Raised when a durable FDE regression contract is invalid."""


def _object(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegressionError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RegressionError(f"JSON root must be an object: {path}")
    return payload


def _finite(payload: Mapping[str, object], key: str, context: str) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RegressionError(f"invalid {key!r} in {context}") from error
    if not math.isfinite(value):
        raise RegressionError(f"nonfinite {key!r} in {context}")
    return value


def _hex_identifier(payload: Mapping[str, object], key: str,
                    length: int, context: str) -> str:
    value = payload.get(key)
    if (not isinstance(value, str) or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)):
        raise RegressionError(f"invalid {key!r} in {context}")
    return value


def _load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RegressionError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_band_folding_reference(payload: Mapping[str, object]) -> Dict[str, object]:
    tolerance = _finite(payload, "tolerance_ry", "LiH band-folding reference")
    error = _finite(
        payload, "maximum_absolute_error_ry", "LiH band-folding reference"
    )
    if int(payload.get("schema_version", 0)) != 1:
        raise RegressionError("unsupported LiH band-folding reference schema")
    if payload.get("passed") is not True or tolerance <= 0.0 or error > tolerance:
        raise RegressionError(
            f"LiH band-folding reference fails: {error:.6g} > {tolerance:.6g} Ry"
        )
    spins = payload.get("spins")
    if not isinstance(spins, list) or [entry.get("spin") for entry in spins] != [0, 1]:
        raise RegressionError("LiH reference must contain ordered alpha/beta results")
    return {"passed": True, "maximum_absolute_error_ry": error,
            "tolerance_ry": tolerance}


def validate_mpi_reference(payload: Mapping[str, object]) -> Dict[str, object]:
    if int(payload.get("schema", 0)) != 1 or payload.get("consistent") is not True:
        raise RegressionError("MPI scaling reference is not a consistent schema-1 result")
    maximum_energy = _finite(payload, "max_energy_delta_ry", "MPI reference")
    energy_tolerance = _finite(payload, "energy_tolerance_ry", "MPI reference")
    maximum_drho = _finite(payload, "max_drho_delta", "MPI reference")
    drho_tolerance = _finite(payload, "drho_tolerance", "MPI reference")
    if maximum_energy > energy_tolerance or maximum_drho > drho_tolerance:
        raise RegressionError("MPI reference exceeds its numerical consistency budget")
    cases = payload.get("cases")
    expected = ["1rank_1thread", "4rank_1thread", "20rank_4thread", "40rank_4thread"]
    if not isinstance(cases, list) or [case.get("layout") for case in cases] != expected:
        raise RegressionError("MPI reference does not contain the four maintained layouts")
    return {"passed": True, "maximum_energy_delta_ry": maximum_energy,
            "maximum_drho_delta": maximum_drho, "case_count": len(cases)}


def validate_cuda_reference(payload: Mapping[str, object]) -> Dict[str, object]:
    if (int(payload.get("schema_version", 0)) != 1
            or payload.get("consistent") is not True
            or payload.get("slurm_state") != "COMPLETED"):
        raise RegressionError("CUDA reference is not a completed consistent schema-1 result")
    energy_tolerance = _finite(payload, "energy_tolerance_ry", "CUDA reference")
    component_tolerance = _finite(
        payload, "component_energy_tolerance_ry", "CUDA reference"
    )
    cases = payload.get("cases")
    expected = {
        "thomas_fermi_uniform",
        "pw91k_uniform",
        "pw91k_nonuniform",
        "revapbek_uniform",
        "revapbek_nonuniform",
    }
    if not isinstance(cases, list) or {case.get("label") for case in cases} != expected:
        raise RegressionError("CUDA reference lacks a maintained KEDF/grid case")
    maximum_energy = 0.0
    maximum_component = 0.0
    for case in cases:
        label = str(case.get("label", "unknown"))
        energy = _finite(case, "absolute_total_energy_delta_ry", label)
        component = _finite(case, "absolute_nake_delta_ry", label)
        speedup = _finite(case, "cpu_over_gpu_wall_speedup", label)
        if speedup <= 0.0:
            raise RegressionError(f"invalid wall-time ratio in {label}")
        maximum_energy = max(maximum_energy, energy)
        maximum_component = max(maximum_component, component)
    if maximum_energy > energy_tolerance or maximum_component > component_tolerance:
        raise RegressionError("CUDA reference exceeds its CPU/GPU agreement budget")
    return {"passed": True, "case_count": len(cases),
            "maximum_total_energy_delta_ry": maximum_energy,
            "maximum_nake_delta_ry": maximum_component}


def validate_performance_reference(payload: Mapping[str, object]) -> Dict[str, object]:
    if int(payload.get("schema_version", 0)) != 1:
        raise RegressionError("unsupported workflow-performance reference schema")
    if payload.get("controlled_speedup_comparison") is not False:
        raise RegressionError("workflow snapshot must not claim a controlled speedup")
    records = payload.get("records")
    if not isinstance(records, list) or {row.get("label") for row in records} != {
        "production", "tight_scf"
    }:
        raise RegressionError("workflow snapshot must contain production and tight_scf")
    for record in records:
        label = str(record.get("label", "unknown"))
        if (_finite(record, "total_profiled_seconds", label) <= 0.0
                or _finite(record, "request_wall_seconds", label) <= 0.0
                or int(record.get("subsystem_calls", 0)) <= 0
                or int(record.get("scf_iterations", 0)) <= 0):
            raise RegressionError(f"invalid workflow performance record: {label}")
        reuse = _finite(record, "session_reuse_fraction", label)
        if not 0.0 <= reuse <= 1.0:
            raise RegressionError(f"invalid session reuse fraction: {label}")
    return {"passed": True, "record_count": len(records)}


def validate_manifest(source_root: Path, manifest_path: Path) -> Dict[str, object]:
    manifest = _object(manifest_path)
    if int(manifest.get("schema_version", 0)) != 1:
        raise RegressionError("unsupported FDE regression manifest schema")
    required_paths = manifest.get("required_paths")
    if not isinstance(required_paths, list) or not required_paths:
        raise RegressionError("regression manifest requires a nonempty path inventory")
    missing = [path for path in required_paths
               if not (source_root / str(path)).is_file()]
    if missing:
        raise RegressionError("missing durable regression files: " + ", ".join(missing))

    references = manifest.get("references")
    if not isinstance(references, dict):
        raise RegressionError("regression manifest lacks references")

    molecular = references.get("molecular_diabatic")
    if not isinstance(molecular, dict):
        raise RegressionError("regression manifest lacks molecular reference")
    compare_path = source_root / str(molecular["comparison_tool"])
    compare_module = _load_python_module(compare_path)
    budget_path = source_root / str(molecular["budget"])
    pes_path = source_root / str(molecular["pes"])
    convergence_path = source_root / str(molecular["convergence"])
    comparison = compare_module.compare(
        _object(budget_path), pes_path, pes_path, convergence_path
    )
    if not comparison.get("passed") or any(comparison["maxima"].values()):
        raise RegressionError("molecular reference does not pass exact self-comparison")
    molecular_provenance = _object(source_root / str(molecular["provenance"]))
    if int(molecular_provenance.get("schema_version", 0)) != 1:
        raise RegressionError("unsupported molecular provenance schema")
    _hex_identifier(molecular_provenance, "abacus_source_commit", 40,
                    "molecular provenance")
    _hex_identifier(molecular_provenance, "abacus_binary_sha256", 64,
                    "molecular provenance")
    molecular_validation = molecular_provenance.get("validation")
    if (not isinstance(molecular_validation, dict)
            or molecular_validation.get("scientific_error_budget_passed") is not True):
        raise RegressionError("molecular provenance lacks accepted tight validation")

    periodic = references.get("periodic_band_folding")
    mpi = references.get("mpi_scaling")
    cuda = references.get("cuda_consistency")
    performance = references.get("workflow_performance")
    if not all(isinstance(item, dict) for item in (periodic, mpi, cuda, performance)):
        raise RegressionError("regression manifest has an incomplete reference inventory")
    periodic_provenance = _object(source_root / str(periodic["provenance"]))
    if int(periodic_provenance.get("schema_version", 0)) != 1:
        raise RegressionError("unsupported periodic provenance schema")
    _hex_identifier(periodic_provenance, "abacus_source_commit", 40,
                    "periodic provenance")
    _hex_identifier(periodic_provenance, "abacus_binary_sha256", 64,
                    "periodic provenance")

    return {
        "passed": True,
        "required_file_count": len(required_paths),
        "molecular_diabatic": {
            "passed": True,
            "point_count": comparison["point_count"],
            "maximum_delta": max(comparison["maxima"].values()),
        },
        "periodic_band_folding": validate_band_folding_reference(
            _object(source_root / str(periodic["result"]))
        ),
        "mpi_scaling": validate_mpi_reference(
            _object(source_root / str(mpi["result"]))
        ),
        "cuda_consistency": validate_cuda_reference(
            _object(source_root / str(cuda["result"]))
        ),
        "workflow_performance": validate_performance_reference(
            _object(source_root / str(performance["result"]))
        ),
    }


def run_python_suites(source_root: Path,
                      suites: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    results = []
    for suite in suites:
        label = str(suite["label"])
        command = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(suite["start_directory"]),
            "-p",
            str(suite.get("pattern", "test_*.py")),
        ]
        completed = subprocess.run(command, cwd=source_root, check=False)
        results.append({"label": label, "command": command,
                        "returncode": completed.returncode,
                        "passed": completed.returncode == 0})
    return results


def run_native_ctest(build_directory: Path) -> Dict[str, object]:
    ctest = shutil.which("ctest")
    if ctest is None:
        raise RegressionError("ctest is required when --build-directory is used")
    if not (build_directory / "CTestTestfile.cmake").is_file():
        raise RegressionError(
            f"build directory has no CTestTestfile.cmake: {build_directory}"
        )
    command = [ctest, "--test-dir", str(build_directory),
               "--output-on-failure", "-R", "^MODULE_FDE_"]
    completed = subprocess.run(command, check=False)
    return {"command": command, "returncode": completed.returncode,
            "passed": completed.returncode == 0}


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_root)
    parser.add_argument("--manifest", type=Path,
                        default=Path("tools/fde/regression_manifest.json"))
    parser.add_argument("--skip-python-tests", action="store_true")
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    source_root = arguments.source_root.resolve(strict=True)
    manifest_path = arguments.manifest
    if not manifest_path.is_absolute():
        manifest_path = source_root / manifest_path

    try:
        reference_report = validate_manifest(source_root, manifest_path)
        manifest = _object(manifest_path)
        python_results = [] if arguments.skip_python_tests else run_python_suites(
            source_root, manifest["python_suites"]
        )
        native_result = None
        if arguments.build_directory is not None:
            native_result = run_native_ctest(arguments.build_directory.resolve())
    except (KeyError, OSError, RegressionError) as error:
        parser.error(str(error))

    passed = (reference_report["passed"]
              and all(result["passed"] for result in python_results)
              and (native_result is None or native_result["passed"]))
    report = {
        "schema_version": 1,
        "passed": passed,
        "source_root": str(source_root),
        "manifest": str(manifest_path),
        "references": reference_report,
        "python_suites": python_results,
        "native_ctest": native_result,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        write_json_atomic(arguments.output.resolve(), report)
    sys.stdout.write(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
