"""Tests for Discord bot lifecycle control."""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner.discord_bot_control import (
    discover_discord_bot_pids,
    ensure_discord_bot_for_handoff_root,
    ensure_discord_bot_running,
    record_discord_bot_site_config,
    stop_discord_bot,
)
from syndiff_pipeline.template_runner import logs
from syndiff_pipeline.template_runner.workspace import record_deployment_path
from tests.site_config import write_site_config, write_site_deployment


def _enabled_bot_setup(base: Path) -> tuple[Path, Path, Path]:
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
    deploy_path = base / "deployment.yaml"
    deploy_path.write_text(
        deploy_path.read_text(encoding="utf-8")
        + "discord_bot_token: token\n"
        + "discord_channel_id: '123'\n",
        encoding="utf-8",
    )
    return cfg_path, handoff, deploy_path


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
            result = ensure_discord_bot_running(
                base / "deployment.yaml",
                site_config_path=cfg_path,
            )
            self.assertFalse(result.enabled)
            self.assertIsNone(result.pid)

    def test_spawns_when_enabled_and_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg_path, handoff, deploy_path = _enabled_bot_setup(base)
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
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.discover_discord_bot_pids",
                return_value=[],
            ):
                result = ensure_discord_bot_running(
                    deploy_path,
                    site_config_path=cfg_path,
                )
            spawn.assert_called_once()
            self.assertTrue(result.enabled)
            self.assertTrue(result.spawned)
            self.assertEqual(result.pid, 4242)

    def test_records_site_config_and_restarts_from_handoff_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg_path, handoff, deploy_path = _enabled_bot_setup(base)
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
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.discover_discord_bot_pids",
                return_value=[],
            ):
                ensure_discord_bot_running(deploy_path, site_config_path=cfg_path)
                record_deployment_path(handoff, deploy_path)
                result = ensure_discord_bot_for_handoff_root(handoff)
            self.assertTrue(logs.discord_bot_site_config_path(handoff).is_file())
            self.assertIsNotNone(result)
            self.assertTrue(result.spawned)

    def test_concurrent_ensure_spawns_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg_path, handoff, deploy_path = _enabled_bot_setup(base)
            spawn_count = 0
            spawn_lock = threading.Lock()

            def counting_spawn(*args, **kwargs):
                nonlocal spawn_count
                with spawn_lock:
                    spawn_count += 1
                return 5000 + spawn_count

            def wait_side_effect(*args, **kwargs):
                return True

            read_pids = iter([None, 5001, 5001])

            def read_pid_side_effect(*args, **kwargs):
                try:
                    return next(read_pids)
                except StopIteration:
                    return 5001

            def is_process_alive_side_effect(pid):
                return pid == 5001

            with mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.spawn_detached_discord_bot",
                side_effect=counting_spawn,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.wait_for_discord_bot",
                side_effect=wait_side_effect,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.daemon.read_pid",
                side_effect=read_pid_side_effect,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.daemon.is_process_alive",
                side_effect=is_process_alive_side_effect,
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.discover_discord_bot_pids",
                return_value=[],
            ):
                barrier = threading.Barrier(2)

                def worker():
                    barrier.wait()
                    ensure_discord_bot_running(deploy_path, site_config_path=cfg_path)

                threads = [threading.Thread(target=worker) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(spawn_count, 1)

    def test_record_discord_bot_site_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            cfg_path = base / "config.yaml"
            cfg_path.write_text("x: 1\n", encoding="utf-8")
            record_discord_bot_site_config(handoff, cfg_path)
            text = logs.discord_bot_site_config_path(handoff).read_text(encoding="utf-8").strip()
            self.assertEqual(text, str(cfg_path.resolve()))

    def test_stop_removes_stale_pid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir(parents=True)
            pid_path = handoff / "discord_bot.pid"
            pid_path.write_text("99999", encoding="utf-8")
            with mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.discover_discord_bot_pids",
                return_value=[],
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.daemon.is_process_alive",
                return_value=False,
            ):
                stopped = stop_discord_bot(handoff)
            self.assertTrue(stopped)
            self.assertFalse(pid_path.is_file())

    def test_stop_terminates_discovered_orphans(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir(parents=True)
            with mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control.discover_discord_bot_pids",
                side_effect=[[111, 222], []],
            ), mock.patch(
                "syndiff_pipeline.template_runner.discord_bot_control._terminate_pids",
            ) as terminate:
                stopped = stop_discord_bot(handoff)
            self.assertTrue(stopped)
            terminate.assert_called_once()
            self.assertEqual(terminate.call_args.args[0], [111, 222])

    def test_discover_discord_bot_pids_empty_when_no_proc(self):
        with mock.patch(
            "syndiff_pipeline.template_runner.discord_bot_control.Path"
        ) as path_cls:
            path_cls.return_value.is_dir.return_value = False
            self.assertEqual(discover_discord_bot_pids("/tmp/handoff"), [])


if __name__ == "__main__":
    unittest.main()
