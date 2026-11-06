import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fde_session import FdeSessionError, FdeSessionProcess


FAKE_WORKER = """\
#!/usr/bin/env python3
import sys

print("FDE_SESSION_READY 1", flush=True)
request_index = 0
for line in sys.stdin:
    fields = line.split()
    if fields == ["STOP"]:
        print("FDE_SESSION_STOPPED", flush=True)
        break
    if len(fields) == 3 and fields[0] == "RUN":
        if fields[1] == "fail":
            print("FDE_SESSION_ERROR fail requested failure", flush=True)
            break
        mode = ("density_seed" if request_index == 0
                else "resident_ao_density_matrix_and_orbitals")
        print("FDE_SESSION_DONE " + fields[1] + " " + mode + " "
              + str(request_index), flush=True)
        request_index += 1
"""


class FdeSessionProcessTest(unittest.TestCase):
    def test_group_signal_permission_race_reaps_an_exiting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FdeSessionProcess(["unused"], Path(directory), os.environ, 0.1, 0.1)
            session.process = mock.Mock(pid=12345)
            session.process.wait.return_value = 2
            with mock.patch("fde_session.os.killpg", side_effect=PermissionError(1, "exiting group")):
                session._terminate_owned_process(False)
            session.process.wait.assert_called_once_with(timeout=0.1)

    def test_group_signal_permission_failure_is_not_hidden_for_a_live_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FdeSessionProcess(["unused"], Path(directory), os.environ, 0.1, 0.1)
            session.process = mock.Mock(pid=12345)
            session.process.wait.side_effect = subprocess.TimeoutExpired(["unused"], 0.1)
            error = PermissionError(1, "live group")
            with mock.patch("fde_session.os.killpg", side_effect=error):
                with self.assertRaises(PermissionError) as raised:
                    session._terminate_owned_process(True)
            self.assertIs(raised.exception, error)
            session.process.wait.assert_called_once_with(timeout=0.1)

    def test_start_timeout_cleans_owned_processes_and_preserves_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "child.pid"
            worker = root / "blocked.py"
            worker.write_text(
                "import pathlib,subprocess,sys,time\n"
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                "pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n"
                "print('startup diagnostic before READY',flush=True)\n"
                "time.sleep(60)\n")
            session = FdeSessionProcess([sys.executable, str(worker), str(pid_file)],
                                        root / "session", os.environ, 0.5, 0.1)
            try:
                with self.assertRaisesRegex(FdeSessionError, "startup diagnostic before READY"):
                    with session:
                        self.fail("worker never sent READY")
                self.assertIsNone(session.process)
                self.assertIsNone(session.log)
                pid = int(pid_file.read_text())
                state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                       text=True, capture_output=True).stdout.strip()
                self.assertTrue(not state or state.startswith("Z"), state)
            finally:
                # Also clean up the intentionally broken pre-fix implementation.
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if session.process is not None and session.process.poll() is None:
                    session.process.kill()
                session.close()

    def test_missing_command_start_failure_closes_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FdeSessionProcess(["/fde-nonexistent-command"], Path(directory),
                                        os.environ, 0.1, 0.1)
            try:
                with self.assertRaises(FileNotFoundError):
                    session.start()
                self.assertIsNone(session.process)
                self.assertIsNone(session.log)
            finally:
                session.close()

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
            self.assertEqual(first["warm_start_mode"], "density_seed")
            self.assertEqual(
                second["warm_start_mode"],
                "resident_ao_density_matrix_and_orbitals")
            self.assertEqual(second["abacus_ionic_step"], 1)
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
