"""Tests for NFS / host mismatch warnings from scheduler_control."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.common.orchestration.scheduler_control import (
    ensure_daemon_running,
    warn_if_daemon_host_mismatch,
)


class TestHostMismatch(unittest.TestCase):
    def test_warn_if_daemon_host_mismatch_prints_sqlite_wal_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            buf = io.StringIO()
            pid_path = logs.daemon_pid_path(handoff)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            daemon.write_process_identity(pid_path, 424242, host="compute-node.cluster")
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.daemon.local_hostname",
                return_value="login-node.local",
            ):
                with redirect_stderr(buf):
                    warn_if_daemon_host_mismatch(handoff)

            err = buf.getvalue()
            self.assertIn("SQLite WAL", err)
            self.assertIn("compute-node.cluster", err)
            self.assertIn("login-node.local", err)

    def test_matching_host_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            pid_path = logs.daemon_pid_path(tmp)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            daemon.write_process_identity(pid_path, 1, host="login-node.local")
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.daemon.local_hostname",
                return_value="login-node.local",
            ):
                with redirect_stderr(buf):
                    warn_if_daemon_host_mismatch(tmp)
            self.assertEqual(buf.getvalue(), "")

    def test_ensure_daemon_running_errors_on_remote_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon_is_alive",
                return_value=True,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.get_supervisor_host",
                return_value="submit01",
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.daemon.local_hostname",
                return_value="login02",
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control._supervisor_pid_identity",
                return_value=("submit01", 99),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_daemon_running(tmp)
            self.assertIn("submit01", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
