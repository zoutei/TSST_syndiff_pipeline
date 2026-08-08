"""Tests for in-process Discord bot configuration helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
    DiscordBotStatus,
    cleanup_legacy_detached_bots,
    discover_workspace_bot_pids,
    discord_bot_status,
    record_discord_bot_site_config,
    should_start_in_process_bot,
    stop_all_workspace_discord_bots,
)
from syndiff_pipeline.common.orchestration.workspace import record_deployment_path
from tests.site_fixtures import write_site_deployment


def _write_pipeline_config(path: Path, *, bot_enabled: bool = True) -> None:
    path.write_text(
        "notifications:\n"
        "  enabled: false\n"
        "  bot:\n"
        f"    enabled: {str(bot_enabled).lower()}\n",
        encoding="utf-8",
    )


def _write_deployment_with_bot(
    config_dir: Path,
    *,
    workspace_root: str,
    data_root: str,
) -> Path:
    path = config_dir / "deployment.yaml"
    path.write_text(
        "\n".join(
            [
                f"workspace_root: {workspace_root}",
                f"data_root: {data_root}",
                "discord_bot_token: token",
                "discord_channel_id: '123'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class TestDiscordBotConfig(unittest.TestCase):
    def test_should_start_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir()
            deploy = _write_deployment_with_bot(
                base,
                workspace_root=str(handoff),
                data_root=str(base / "data"),
            )
            cfg = base / "pipeline.yaml"
            _write_pipeline_config(cfg, bot_enabled=True)
            record_deployment_path(handoff, deploy)
            record_discord_bot_site_config(handoff, cfg)
            with mock.patch.dict("sys.modules", {"discord": mock.Mock()}):
                should, reason, path = should_start_in_process_bot(handoff)
            self.assertTrue(should)
            self.assertIsNone(reason)
            self.assertEqual(path, deploy.resolve())

    def test_should_not_start_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir()
            deploy = _write_deployment_with_bot(
                base,
                workspace_root=str(handoff),
                data_root=str(base / "data"),
            )
            cfg = base / "pipeline.yaml"
            _write_pipeline_config(cfg, bot_enabled=False)
            record_deployment_path(handoff, deploy)
            record_discord_bot_site_config(handoff, cfg)
            should, reason, _path = should_start_in_process_bot(handoff)
            self.assertFalse(should)
            self.assertEqual(reason, "disabled")

    def test_status_expected_in_process_when_daemon_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            deploy = _write_deployment_with_bot(
                base,
                workspace_root=str(base / "handoff"),
                data_root=str(base / "data"),
            )
            cfg = base / "pipeline.yaml"
            _write_pipeline_config(cfg, bot_enabled=True)
            with mock.patch.dict("sys.modules", {"discord": mock.Mock()}):
                st = discord_bot_status(deploy, site_config_path=cfg, daemon_alive=True)
            self.assertIsInstance(st, DiscordBotStatus)
            self.assertTrue(st.enabled)
            self.assertTrue(st.expected_in_process)

    def test_cleanup_legacy_detached_bots_terminates_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.discover_workspace_bot_pids",
                return_value=[4242],
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.daemon.terminate_process_tree",
            ) as term, mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.daemon.read_process_identity",
                return_value=(None, None),
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.daemon.is_process_alive",
                return_value=False,
            ):
                n = cleanup_legacy_detached_bots(tmp)
            self.assertEqual(n, 1)
            term.assert_called()

    def test_stop_all_uses_workspace_discover(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.discover_workspace_bot_pids",
                return_value=[9001],
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.daemon.terminate_process_tree",
            ) as term, mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.daemon.read_process_identity",
                return_value=(None, None),
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.daemon.is_process_alive",
                side_effect=lambda pid: False,
            ):
                n = stop_all_workspace_discord_bots(tmp)
            self.assertEqual(n, 1)
            term.assert_called()

    def test_discover_workspace_bot_pids_matches_without_detached(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir()
            deploy = _write_deployment_with_bot(
                Path(tmp),
                workspace_root=str(handoff),
                data_root=str(Path(tmp) / "data"),
            )
            cmdline = (
                b"python\0-m\0syndiff_pipeline.template_creation.orchestration.discord_bot"
                b"\0--deployment\0" + str(deploy).encode() + b"\0"
            )
            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.discord_bot_control.Path"
            ) as path_cls:
                proc = mock.MagicMock()
                proc.is_dir.return_value = True
                entry = mock.MagicMock()
                entry.name = "55555"
                entry.__truediv__ = lambda self, name: mock.Mock(
                    read_bytes=mock.Mock(return_value=cmdline)
                )
                proc.iterdir.return_value = [entry]

                def path_side_effect(arg):
                    if str(arg) == "/proc":
                        return proc
                    return Path(arg)

                path_cls.side_effect = path_side_effect
                with mock.patch(
                    "syndiff_pipeline.template_creation.orchestration.discord_bot_control.load_workspace_root_from_deployment",
                    return_value=str(handoff.resolve()),
                ), mock.patch(
                    "syndiff_pipeline.template_creation.orchestration.discord_bot_control.normalize_workspace_root",
                    return_value=str(handoff.resolve()),
                ), mock.patch(
                    "syndiff_pipeline.template_creation.orchestration.discord_bot_control.daemon.is_process_alive",
                    return_value=True,
                ):
                    pids = discover_workspace_bot_pids(handoff)
            self.assertEqual(pids, [55555])


if __name__ == "__main__":
    unittest.main()
