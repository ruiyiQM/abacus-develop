import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("fetch_default_resources.py")
SPEC = importlib.util.spec_from_file_location("fetch_default_resources", MODULE_PATH)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class FetchDefaultResourcesTest(unittest.TestCase):
    def test_copies_and_verifies_pinned_local_resource(self):
        payload = b"pinned resource\n"
        checksum = hashlib.sha256(payload).hexdigest()
        manifest = {
            "schema_version": 1,
            "source_repository": "https://github.com/example/resources",
            "source_commit": "0123456789abcdef",
            "resources": [{
                "kind": "orbitals",
                "element": "H",
                "path": "basis/H.orb",
                "sha256": checksum,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "basis"
            source.mkdir(parents=True)
            (source / "H.orb").write_bytes(payload)
            destination = root / "destination"

            installed = list(fetch.install_resources(
                manifest, destination, root / "source"))

            self.assertEqual(installed, [destination / "orbitals" / "H.orb"])
            self.assertEqual(installed[0].read_bytes(), payload)
            written = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(written["source_commit"], "0123456789abcdef")

    def test_refuses_to_replace_mismatched_existing_resource(self):
        expected = hashlib.sha256(b"expected").hexdigest()
        manifest = {
            "schema_version": 1,
            "source_repository": "https://github.com/example/resources",
            "source_commit": "0123456789abcdef",
            "resources": [{
                "kind": "pseudopotentials",
                "element": "F",
                "path": "pseudo/F.upf",
                "sha256": expected,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "destination" / "pseudopotentials" / "F.upf"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"do not overwrite")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                list(fetch.install_resources(manifest, root / "destination",
                                             root / "source"))
            self.assertEqual(target.read_bytes(), b"do not overwrite")

    def test_rejects_parent_traversal_in_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "source_repository": "https://github.com/example/resources",
                "source_commit": "0123456789abcdef",
                "resources": [{
                    "kind": "orbitals",
                    "path": "../outside.orb",
                    "sha256": "0" * 64,
                }],
            }))
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                fetch.load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
