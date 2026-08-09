#!/usr/bin/env python3
"""Fetch and verify the pinned SG15 PBE/DZP resources for this example."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "default_resources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported default-resource manifest schema")
    if not manifest.get("source_repository") or not manifest.get("source_commit"):
        raise ValueError("default-resource manifest has incomplete provenance")
    resources = manifest.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("default-resource manifest has no resources")
    for resource in resources:
        if resource.get("kind") not in ("pseudopotentials", "orbitals"):
            raise ValueError("default-resource manifest contains an invalid kind")
        source_path = Path(resource.get("path", ""))
        if (not source_path.name or source_path.is_absolute()
                or ".." in source_path.parts):
            raise ValueError("default-resource manifest contains an unsafe path")
        checksum = resource.get("sha256", "")
        if len(checksum) != 64 or any(character not in "0123456789abcdef"
                                      for character in checksum):
            raise ValueError("default-resource manifest contains an invalid SHA-256")
    return manifest


def raw_url(repository: str, commit: str, source_path: str) -> str:
    prefix = repository.rstrip("/")
    if prefix.startswith("https://github.com/"):
        prefix = prefix.replace("https://github.com/",
                                "https://raw.githubusercontent.com/", 1)
    return f"{prefix}/{commit}/{quote(source_path)}"


def copy_stream(source: Any, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def install_resources(manifest: Dict[str, Any], destination: Path,
                      source_root: Optional[Path] = None,
                      force: bool = False) -> Iterable[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    installed = []
    for resource in manifest["resources"]:
        source_path = Path(resource["path"])
        target = destination / resource["kind"] / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == resource["sha256"]:
            installed.append(target)
            continue
        if target.exists() and not force:
            raise ValueError(f"checksum mismatch (use --force to replace): {target}")

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=str(target.parent), delete=False) as stream:
                temporary_path = Path(stream.name)
            if source_root is not None:
                source = source_root / source_path
                if not source.is_file():
                    raise FileNotFoundError(f"resource is missing from source root: {source}")
                with source.open("rb") as input_stream:
                    copy_stream(input_stream, temporary_path)
            else:
                url = raw_url(manifest["source_repository"],
                              manifest["source_commit"], resource["path"])
                request = Request(url, headers={"User-Agent": "ABACUS-FDE-example/1"})
                with urlopen(request) as input_stream:
                    copy_stream(input_stream, temporary_path)
            actual = sha256(temporary_path)
            if actual != resource["sha256"]:
                raise ValueError(
                    f"checksum mismatch for {source_path.name}: "
                    f"expected {resource['sha256']}, got {actual}")
            temporary_path.replace(target)
            temporary_path = None
            installed.append(target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    installed_manifest = dict(manifest)
    installed_manifest["installed_files"] = [str(path.resolve()) for path in installed]
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(installed_manifest, indent=2) + "\n",
                             encoding="utf-8")
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "resources")
    parser.add_argument(
        "--source-root", type=Path,
        help="copy from a local ABACUS-orbitals checkout instead of downloading")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing file whose checksum is wrong")
    arguments = parser.parse_args()

    manifest = load_manifest()
    source_root = arguments.source_root.resolve() if arguments.source_root else None
    if source_root is not None and not source_root.is_dir():
        parser.error(f"source root does not exist: {source_root}")
    try:
        installed = list(install_resources(manifest, arguments.destination.resolve(),
                                           source_root, arguments.force))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Verified {len(installed)} resources in {arguments.destination.resolve()}")
    print(f"Pinned source commit: {manifest['source_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
