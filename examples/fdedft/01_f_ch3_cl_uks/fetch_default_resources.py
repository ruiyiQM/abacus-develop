#!/usr/bin/env python3
"""Fetch and verify the pinned SG15 PBE/DZP resources for this example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "default_resources.json"
TOOL_PATH = ROOT.parents[2] / "tools/fde/fetch_example_resources.py"
SPEC = importlib.util.spec_from_file_location("fetch_example_resources", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import shared resource tool: {TOOL_PATH}")
_shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_shared)

sha256 = _shared.sha256
load_manifest = _shared.load_manifest
raw_url = _shared.raw_url
copy_stream = _shared.copy_stream
install_resources = _shared.install_resources


def main() -> int:
    return _shared.main(
        sys.argv[1:], DEFAULT_MANIFEST, ROOT / "resources")


if __name__ == "__main__":
    raise SystemExit(main())
