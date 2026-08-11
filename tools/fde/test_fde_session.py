import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fde_session import FdeSessionError, FdeSessionProcess


FAKE_WORKER = """\
#!/usr/bin/env python3
import sys

print("FDE_SESSION_READY 1", flush=True)
for line in sys.stdin:
    fields = line.split()
    if fields == ["STOP"]:
        print("FDE_SESSION_STOPPED", flush=True)
        break
    if len(fields) == 3 and fields[0] == "RUN":
        if fields[1] == "fail":
            print("FDE_SESSION_ERROR fail requested failure", flush=True)
            break
        print("FDE_SESSION_DONE " + fields[1], flush=True)
"""


class FdeSessionProcessTest(unittest.TestCase):
    def make_worker(self, root: Path) -> Path:
        worker = root / "fake_worker.py"
        worker.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8")
        worker.chmod(0o755)
        return worker

    def test_reuses_one_event_driven_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = self.make_worker(root)
            session = FdeSessionProcess(
                [str(worker)], root / "session", os.environ,
                startup_timeout_seconds=5.0, stop_timeout_seconds=5.0)
            session.start()
            first = session.run("one", (root / "one.config").resolve())
            second = session.run("two", (root / "two.config").resolve())
            session.close()
            self.assertFalse(first["session_reused"])
            self.assertTrue(second["session_reused"])
            self.assertEqual(first["session_pid"], second["session_pid"])
            log = (root / "session" / "fde_session.log").read_text(
                encoding="utf-8")
            self.assertEqual(log.count("FDE_SESSION_READY"), 1)
            self.assertIn("FDE_SESSION_STOPPED", log)

    def test_propagates_protocol_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = self.make_worker(root)
            session = FdeSessionProcess(
                [str(worker)], root / "session", os.environ,
                startup_timeout_seconds=5.0, stop_timeout_seconds=5.0)
            session.start()
            with self.assertRaisesRegex(FdeSessionError, "requested failure"):
                session.run("fail", (root / "fail.config").resolve())
            session.close()


if __name__ == "__main__":
    unittest.main()
