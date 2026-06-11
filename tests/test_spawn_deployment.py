"""Tests for deployment-based detached spawns."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.template_creation.orchestration.discord_bot_control import spawn_detached_discord_bot
from syndiff_pipeline.common.orchestration.scheduler_control import ensure_daemon_running
from syndiff_pipeline.common.orchestration.workspace import record_deployment_path
from tests.site_fixtures import write_site_deployment


class TestSpawnDeploymentArgv(unittest.TestCase):
    def test_spawn_detached_daemon_uses_deployment_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            deploy = base / "deployment.yaml"
            handoff = base / "handoff"
            write_site_deployment(base, workspace_root=str(handoff), data_root=str(base / "data"))
            with mock.patch(
                "syndiff_pipeline.common.orchestration.daemon.subprocess.Popen",
            ) as popen:
                popen.return_value.pid = 1234
                pid = daemon.spawn_detached_daemon(deploy, base / "daemon.log")
            self.assertEqual(pid, 1234)
            cmd = popen.call_args.args[0]
            self.assertIn("--deployment", cmd)
            self.assertIn(str(deploy.resolve()), cmd)
            self.assertNotIn("--handoff-root", cmd)

    def test_spawn_detached_discord_bot_uses_deployment_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            deploy = base / "deployment.yaml"
            handoff = base / "handoff"
            write_site_deployment(base, workspace_root=str(handoff), data_root=str(base / "data"))
            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.subprocess.Popen",
            ) as popen:
                popen.return_value.pid = 5678
                pid = spawn_detached_discord_bot(deploy, base / "bot.log")
            self.assertEqual(pid, 5678)
            cmd = popen.call_args.args[0]
            self.assertIn("--deployment", cmd)
            self.assertIn(str(deploy.resolve()), cmd)
            self.assertNotIn("--config", cmd)

    def test_ensure_daemon_requires_recorded_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir()
            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon_is_alive",
                return_value=False,
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon_is_wedged",
                return_value=False,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_daemon_running(str(handoff))
            self.assertIn("deployment.yaml", str(ctx.exception))

    def test_ensure_daemon_uses_recorded_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir()
            deploy = base / "deployment.yaml"
            write_site_deployment(base, workspace_root=str(handoff), data_root=str(base / "data"))
            record_deployment_path(handoff, deploy)
            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon_is_alive",
                return_value=False,
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon_is_wedged",
                return_value=False,
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.spawn_detached_daemon",
                return_value=4242,
            ) as spawn, mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.wait_for_daemon",
                return_value=True,
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.read_process_identity",
                return_value=(None, 4242),
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.local_hostname",
                return_value="localhost",
            ):
                result = ensure_daemon_running(str(handoff))
            spawn.assert_called_once_with(deploy.resolve(), logs.daemon_log_path(handoff))
            self.assertTrue(result.spawned)


if __name__ == "__main__":
    unittest.main()
