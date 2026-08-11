#!/usr/bin/env python3
"""Event-driven client for one persistent ABACUS FDE worker."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Mapping, Sequence


class FdeSessionError(RuntimeError):
    pass


class FdeSessionProcess:
    """Own one ABACUS process speaking the line-oriented FDE session protocol."""

    def __init__(self,
                 command: Sequence[str],
                 working_directory: Path,
                 environment: Mapping[str, str],
                 startup_timeout_seconds: float,
                 stop_timeout_seconds: float):
        if not command or not all(isinstance(token, str) and token for token in command):
            raise ValueError("FDE session command must be a nonempty string array")
        if startup_timeout_seconds <= 0.0 or stop_timeout_seconds <= 0.0:
            raise ValueError("FDE session timeouts must be positive")
        self.command = list(command)
        self.working_directory = working_directory.resolve()
        self.environment = dict(environment)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.stop_timeout_seconds = float(stop_timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.log = None
        self.reader: threading.Thread | None = None
        self.messages: queue.Queue[str] = queue.Queue()
        self.request_count = 0
        self.startup_seconds = 0.0
        self._lock = threading.Lock()

    def _read_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        assert self.log is not None
        try:
            for line in self.process.stdout:
                self.log.write(line)
                self.log.flush()
                marker = line.strip()
                if marker.startswith("FDE_SESSION_"):
                    self.messages.put(marker)
        finally:
            self.messages.put("FDE_SESSION_EOF")

    def _wait_marker(self,
                     prefix: str,
                     timeout_seconds: float | None) -> str:
        while True:
            try:
                marker = self.messages.get(timeout=timeout_seconds)
            except queue.Empty as error:
                raise FdeSessionError(
                    f"timed out waiting for {prefix} from FDE session in "
                    f"{self.working_directory}") from error
            if marker.startswith("FDE_SESSION_ERROR"):
                raise FdeSessionError(marker)
            if marker == "FDE_SESSION_EOF":
                returncode = None if self.process is None else self.process.poll()
                raise FdeSessionError(
                    f"FDE session exited before {prefix}; returncode={returncode}; "
                    f"see {self.working_directory / 'fde_session.log'}")
            if marker.startswith(prefix):
                return marker

    def start(self) -> None:
        if self.process is not None:
            raise FdeSessionError("FDE session process was already started")
        self.working_directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.working_directory / "fde_session.log").open(
            "w", encoding="utf-8")
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            cwd=self.working_directory,
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.reader = threading.Thread(
            target=self._read_output,
            name=f"fde-session-{self.process.pid}",
            daemon=True,
        )
        self.reader.start()
        self._wait_marker("FDE_SESSION_READY ", self.startup_timeout_seconds)
        self.startup_seconds = time.monotonic() - started

    def run(self, request_id: str, config_path: Path) -> dict[str, object]:
        if (not request_id or any(character.isspace() for character in request_id)
                or not config_path.is_absolute()
                or any(character.isspace() for character in str(config_path))):
            raise ValueError(
                "FDE session request id must be a token and config path absolute")
        with self._lock:
            if self.process is None or self.process.stdin is None:
                raise FdeSessionError("FDE session process is not running")
            if self.process.poll() is not None:
                raise FdeSessionError(
                    f"FDE session process already exited with {self.process.returncode}")
            reused = self.request_count > 0
            request_started = time.monotonic()
            self.process.stdin.write(f"RUN {request_id} {config_path}\n")
            self.process.stdin.flush()
            marker = self._wait_marker(
                f"FDE_SESSION_DONE {request_id}", None)
            fields = marker.split()
            if (len(fields) != 4
                    or fields[:2] != ["FDE_SESSION_DONE", request_id]):
                raise FdeSessionError(
                    f"unexpected FDE session completion marker: {marker}")
            try:
                ionic_step = int(fields[3])
            except ValueError as error:
                raise FdeSessionError(
                    f"invalid ABACUS step in completion marker: {marker}") from error
            if ionic_step != self.request_count:
                raise FdeSessionError(
                    f"out-of-order FDE session completion marker: {marker}")
            warm_start_mode = fields[2]
            expected_mode = ("density_seed" if self.request_count == 0
                             else "resident_ao_density_matrix_and_orbitals")
            if warm_start_mode != expected_mode:
                raise FdeSessionError(
                    f"unexpected FDE warm-start mode: {marker}")
            self.request_count += 1
            return {
                "execution_mode": "persistent_session",
                "session_pid": self.process.pid,
                "session_request_index": self.request_count,
                "session_reused": reused,
                "warm_start_mode": warm_start_mode,
                "abacus_ionic_step": ionic_step,
                "session_startup_seconds": self.startup_seconds if not reused else 0.0,
                "wall_time_seconds": time.monotonic() - request_started,
            }

    def close(self) -> None:
        process = self.process
        if process is None:
            if self.log is not None:
                self.log.close()
                self.log = None
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("STOP\n")
                process.stdin.flush()
                self._wait_marker(
                    "FDE_SESSION_STOPPED", self.stop_timeout_seconds)
            process.wait(timeout=self.stop_timeout_seconds)
        except (BrokenPipeError, FdeSessionError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self.stop_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if self.reader is not None:
                self.reader.join(timeout=self.stop_timeout_seconds)
            if self.log is not None:
                self.log.close()
            self.process = None

    def __enter__(self) -> "FdeSessionProcess":
        self.start()
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        self.close()


def default_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    return environment
