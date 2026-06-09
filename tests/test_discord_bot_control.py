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
    EnsureDiscordBotResult,
    ensure_discord_bot_running,
    stop_discord_bot,
)


class TestDiscordBotControl(unittest.TestCase):
    def test_skips_when_bot_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg_path = base / "config.yaml"
            cfg_path.write_text(
                "data_root: /\nhandoff_root: /\nskycell_wcs_csv: /\n"
                "state_db_path: state.sqlite\n"
                "notifications:\n  enabled: true\n  bot:\n    enabled: false\n",
                encoding="utf-8",
            )
            result = ensure_discord_bot_running(cfg_path)
            self.assertFalse(result.enabled)
            self.assertIsNone(result.pid)

    def test_spawns_when_enabled_and_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg_path = base / "config.yaml"
            cfg_path.write_text(
                "data_root: /\nhandoff_root: /\nskycell_wcs_csv: /\n"
                "state_db_path: state.sqlite\n"
                "notifications:\n  enabled: true\n  bot:\n    enabled: true\n",
                encoding="utf-8",
            )
            (base / "secrets.yaml").write_text(
                "discord_bot_token: token\n"
                "discord_channel_id: '123'\n",
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

    def test_stop_removes_stale_pid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db = base / "state.sqlite"
            db.write_text("", encoding="utf-8")
            pid_path = base / "discord_bot.pid"
            pid_path.write_text("99999", encoding="utf-8")
            with mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.daemon.is_process_alive",
                return_value=False,
            ):
                stopped = stop_discord_bot(db)
            self.assertTrue(stopped)
            self.assertFalse(pid_path.is_file())


if __name__ == "__main__":
    unittest.main()
