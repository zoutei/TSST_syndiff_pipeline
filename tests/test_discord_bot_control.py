"""Tests for Discord bot lifecycle control."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner.discord_bot_control import (
    ensure_discord_bot_for_handoff_root,
    ensure_discord_bot_running,
    stop_discord_bot,
)
from syndiff_pipeline.template_runner import logs
from tests.site_config import write_site_config, write_site_deployment


class TestDiscordBotControl(unittest.TestCase):
    def test_skips_when_bot_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            cfg_path = base / "config.yaml"
            write_site_config(
                cfg_path,
                handoff_root=str(handoff),
                data_root=str(base / "data"),
                notifications_enabled=True,
            )
            cfg_path.write_text(
                cfg_path.read_text(encoding="utf-8")
                + "notifications:\n  enabled: true\n"
                + "  bot:\n    enabled: false\n",
                encoding="utf-8",
            )
            result = ensure_discord_bot_running(cfg_path)
            self.assertFalse(result.enabled)
            self.assertIsNone(result.pid)

    def test_spawns_when_enabled_and_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            cfg_path = base / "config.yaml"
            write_site_config(
                cfg_path,
                handoff_root=str(handoff),
                data_root=str(base / "data"),
                notifications_enabled=True,
            )
            cfg_path.write_text(
                "deployment_file: deployment.yaml\n"
                "stages:\n  mapping: {}\n"
                "notifications:\n  enabled: true\n"
                "  bot:\n    enabled: true\n",
                encoding="utf-8",
            )
            write_site_deployment(
                base,
                handoff_root=str(handoff),
                data_root=str(base / "data"),
            )
            (base / "deployment.yaml").write_text(
                (base / "deployment.yaml").read_text(encoding="utf-8")
                + "discord_bot_token: token\n"
                + "discord_channel_id: '123'\n",
                encoding="utf-8",
            )
            with mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.spawn_detached_discord_bot",
                return_value=4242,
            ) as spawn, mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.wait_for_discord_bot",
                return_value=True,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.discord_bot_is_alive",
                return_value=True,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.daemon.read_pid",
                return_value=4242,
            ):
                result = ensure_discord_bot_running(cfg_path)
            spawn.assert_called_once()
            self.assertTrue(result.enabled)
            self.assertTrue(result.spawned)
            self.assertEqual(result.pid, 4242)

    def test_records_site_config_and_restarts_from_handoff_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            cfg_path = base / "config.yaml"
            write_site_config(
                cfg_path,
                handoff_root=str(handoff),
                data_root=str(base / "data"),
                notifications_enabled=True,
            )
            cfg_path.write_text(
                "deployment_file: deployment.yaml\n"
                "stages:\n  mapping: {}\n"
                "notifications:\n  enabled: true\n"
                "  bot:\n    enabled: true\n",
                encoding="utf-8",
            )
            write_site_deployment(
                base,
                handoff_root=str(handoff),
                data_root=str(base / "data"),
            )
            (base / "deployment.yaml").write_text(
                (base / "deployment.yaml").read_text(encoding="utf-8")
                + "discord_bot_token: token\n"
                + "discord_channel_id: '123'\n",
                encoding="utf-8",
            )
            with mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.spawn_detached_discord_bot",
                return_value=4242,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.wait_for_discord_bot",
                return_value=True,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.discord_bot_is_alive",
                side_effect=[False, True],
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.daemon.read_pid",
                return_value=4242,
            ):
                ensure_discord_bot_running(cfg_path)
                result = ensure_discord_bot_for_handoff_root(handoff)
            self.assertTrue(logs.discord_bot_site_config_path(handoff).is_file())
            self.assertIsNotNone(result)
            self.assertTrue(result.spawned)

    def test_stop_removes_stale_pid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir(parents=True)
            pid_path = handoff / "discord_bot.pid"
            pid_path.write_text("99999", encoding="utf-8")
            with mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.daemon.is_process_alive",
                return_value=False,
            ):
                stopped = stop_discord_bot(handoff)
            self.assertTrue(stopped)
            self.assertFalse(pid_path.is_file())


if __name__ == "__main__":
    unittest.main()
