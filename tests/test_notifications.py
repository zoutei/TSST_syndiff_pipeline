"""Tests for Discord pipeline notifications."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner.notifications import (
    _DISCORD_MAX_CONTENT,
    NotificationConfig,
    NotificationEvents,
    Notifier,
    format_preview_message,
    format_run_started_message,
    format_status_reply_message,
    format_status_reply_messages,
    load_bot_token,
    load_webhook_url,
    parse_notification_config,
    post_discord_webhook,
    resolve_bot_token,
    resolve_channel_id,
    resolve_run_ids_for_status_request,
    resolve_webhook_url,
    send_preview_notification,
    send_run_started_notification,
)
from syndiff_pipeline.template_runner.run_report import (
    format_progress_lines,
    format_run_report,
    format_run_report_messages,
    format_status_grid,
)
from syndiff_pipeline.template_runner.state import (
    STAGE_NAMES,
    PipelineState,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SUCCESS,
)
from syndiff_pipeline.template_runner.targets import Target


class TestNotificationConfig(unittest.TestCase):
    def test_parse_defaults(self):
        cfg = parse_notification_config({})
        self.assertFalse(cfg.enabled)
        self.assertTrue(cfg.events.stage_completed)
        self.assertTrue(cfg.events.run_started)
        self.assertFalse(cfg.bot.enabled)

    def test_load_webhook_from_deployment_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "deployment.yaml").write_text(
                "discord_webhook_url: https://example.com/hook\n", encoding="utf-8"
            )
            url = load_webhook_url(base / "config.yaml", "deployment.yaml")
            self.assertEqual(url, "https://example.com/hook")

    def test_resolve_webhook_falls_back_to_source_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "site"
            run = base / "runs" / "run_a"
            source.mkdir(parents=True)
            run.mkdir(parents=True)
            (source / "deployment.yaml").write_text(
                "discord_webhook_url: https://example.com/from-source\n",
                encoding="utf-8",
            )
            url = resolve_webhook_url(
                config_path=run / "config.yaml",
                deployment_file="deployment.yaml",
                source_config_path=source / "config.yaml",
            )
            self.assertEqual(url, "https://example.com/from-source")

    def test_resolve_bot_token_and_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "deployment.yaml").write_text(
                "discord_bot_token: bot-token\n"
                "discord_channel_id: '999'\n",
                encoding="utf-8",
            )
            self.assertEqual(load_bot_token(base / "config.yaml", "deployment.yaml"), "bot-token")
            self.assertEqual(
                resolve_channel_id(
                    config_path=base / "config.yaml",
                    deployment_file="deployment.yaml",
                    config_channel_id="111",
                ),
                "111",
            )
            self.assertEqual(
                resolve_channel_id(
                    config_path=base / "config.yaml",
                    deployment_file="deployment.yaml",
                ),
                "999",
            )
            self.assertEqual(
                resolve_bot_token(
                    config_path=base / "config.yaml",
                    deployment_file="deployment.yaml",
                ),
                "bot-token",
            )


class TestRunStarted(unittest.TestCase):
    def test_format_run_started_message(self):
        text = format_run_started_message(
            "batch_a",
            run_dir="/runs/batch_a",
            target_labels=["s0020_c3_k3_2020ghq", "s0021_c1_k2_sn"],
            stages=["ps1_download", "downsample"],
        )
        self.assertIn("run_started", text)
        self.assertIn("targets: 2", text)
        self.assertIn("ps1_download", text)
        self.assertNotIn("run_id=batch_a status=", text)

    def test_send_run_started_uses_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            handoff = base

            handoff = base


            db = base / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("r1", "/c", "/t", str(base), [target], ["mapping"])
            cfg_path = base / "config.yaml"
            (base / "deployment.yaml").write_text(
                "discord_webhook_url: https://example.com/hook\n", encoding="utf-8"
            )
            cfg = NotificationConfig(enabled=True)
            with mock.patch(
                "syndiff_pipeline.template_runner.notifications.post_discord_webhook"
            ) as post:
                send_run_started_notification(
                    state,
                    cfg,
                    config_path=cfg_path,
                    run_id="r1",
                    run_dir=base / "runs" / "r1",
                    target_labels=[target.label()],
                    stages=["mapping"],
                    handoff_root=str(handoff),
                    deployment_file="deployment.yaml",
                )
                send_run_started_notification(
                    state,
                    cfg,
                    config_path=cfg_path,
                    run_id="r1",
                    run_dir=base / "runs" / "r1",
                    target_labels=[target.label()],
                    stages=["mapping"],
                    handoff_root=str(handoff),
                    deployment_file="deployment.yaml",
                )
                self.assertEqual(post.call_count, 1)
                self.assertIn("run_started", post.call_args[0][1])


class TestStatusReply(unittest.TestCase):
    def test_resolve_run_ids_prefers_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("run_a", "/c", "/t", str(tmp), [target], ["mapping"])
            state.create_run("run_b", "/c", "/t", str(tmp), [target], ["mapping"])
            state.set_run_status("run_a", "running")
            ids = resolve_run_ids_for_status_request(state, "please show run_b")
            self.assertEqual(ids, ["run_b"])

    def test_resolve_run_ids_active_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("run_a", "/c", "/t", str(tmp), [target], ["mapping"])
            state.set_run_status("run_a", "running")
            ids = resolve_run_ids_for_status_request(state, "status?")
            self.assertEqual(ids, ["run_a"])

    def test_resolve_run_ids_returns_all_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            for name in ("run_a", "run_b", "run_c", "run_d", "run_e"):
                state.create_run(name, "/c", "/t", str(tmp), [target], ["mapping"])
                state.set_run_status(name, "running")
            ids = resolve_run_ids_for_status_request(state, "status?")
            self.assertEqual(ids, ["run_a", "run_b", "run_c", "run_d", "run_e"])

    def test_format_status_reply_includes_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("r1", "/c", "/t", str(tmp), [target], ["mapping"])
            state.set_run_status("r1", "running")
            text = format_status_reply_message(state, ["r1"], str(tmp), handoff_root=str(handoff))
            self.assertIn("run_id=r1", text)
            self.assertIn("status (", text)

    def test_format_status_reply_splits_runs_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            targets = [
                Target(i, 1, 1, 200.0 + i, 50.0, f"t{i:04d}")
                for i in range(30)
            ]
            for run_id in ("batch_no5", "s0020_c3_k3_2020ut_hdr_20260609_141441"):
                state.create_run(
                    run_id,
                    "/c",
                    "/t",
                    str(tmp),
                    targets,
                    list(STAGE_NAMES),
                )
                state.set_run_status(run_id, "running")
            messages = format_status_reply_messages(
                state,
                ["batch_no5", "s0020_c3_k3_2020ut_hdr_20260609_141441"],
                str(tmp),
                handoff_root=str(handoff),
            )
            self.assertGreater(len(messages), 1)
            joined = "\n\n".join(messages)
            self.assertNotIn("truncated", joined)
            self.assertIn("s0020_c3_k3_2020ut_hdr_20260609_141441", joined)
            for message in messages:
                self.assertLessEqual(len(message), _DISCORD_MAX_CONTENT)


class TestRunReport(unittest.TestCase):
    def _seed_run(self, state: PipelineState, run_id: str) -> None:
        targets = [
            Target(22, 3, 3, 228.0, 52.0, "2020dgc"),
            Target(23, 1, 3, 230.0, 53.0, "2020ftl"),
        ]
        state.create_run(
            run_id,
            "/cfg.yaml",
            "/targets.csv",
            "/runs",
            targets,
            list(STAGE_NAMES),
        )
        state.update_stage_status(
            run_id, targets[0].label(), "mapping", STATUS_SUCCESS, finished_at="t"
        )
        state.update_stage_status(
            run_id, targets[0].label(), "ps1_download", STATUS_RUNNING, started_at="t"
        )
        state.update_stage_status(
            run_id, targets[1].label(), "mapping", STATUS_PENDING
        )

    def test_format_status_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            self._seed_run(state, "run_a")
            lines = format_status_grid(state, "run_a")
            self.assertEqual(len(lines), 2)
            self.assertTrue(any("map:succ" in line for line in lines))

    def test_format_progress_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            self._seed_run(state, "run_a")
            state.set_run_status("run_a", "running")
            lines = format_progress_lines(state, "run_a", "/runs")
            self.assertTrue(any(line.startswith("run_id=run_a") for line in lines))

    def test_format_run_report_truncates_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            self._seed_run(state, "run_a")
            text = format_run_report(
                state,
                "run_a",
                "/runs",
                header="[run_a] test",
                max_chars=120,
            )
            self.assertIn("[run_a] test", text)
            self.assertLessEqual(len(text), 120)

    def test_format_run_report_messages_splits_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp)

            db = Path(tmp) / "pipeline_state.sqlite"
            state = PipelineState(db)
            targets = [
                Target(i, 1, 1, 200.0 + i, 50.0, f"t{i:04d}")
                for i in range(25)
            ]
            state.create_run(
                "run_big",
                "/c",
                "/t",
                str(tmp),
                targets,
                list(STAGE_NAMES),
            )
            state.set_run_status("run_big", "running")
            messages = format_run_report_messages(
                state,
                "run_big",
                str(tmp),
                header="[run_big] status (test)",
                max_chars=500,
            )
            self.assertGreater(len(messages), 1)
            self.assertIn("status grid (continued)", messages[1])
            for message in messages:
                self.assertLessEqual(len(message), 500)
            self.assertNotIn("truncated", "\n\n".join(messages))


class TestPreview(unittest.TestCase):
    def test_format_preview_includes_test_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            handoff = base

            handoff = base


            db = base / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("r1", "/c", "/t", str(base), [target], ["mapping"])
            state.set_run_status("r1", "running")
            text = format_preview_message(state, "r1", str(base))
            self.assertIn("[TEST]", text)
            self.assertIn("run_id=r1", text)

    def test_send_preview_skips_dedup_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            handoff = base

            handoff = base


            db = base / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("r1", "/c", "/t", str(base), [target], ["mapping"])
            cfg_path = base / "config.yaml"
            cfg_path.write_text("data_root: /\n", encoding="utf-8")
            (base / "deployment.yaml").write_text(
                "discord_webhook_url: https://example.com/hook\n", encoding="utf-8"
            )
            ctx = mock.Mock()
            ctx.run_id = "r1"
            ctx.run_dir = base
            ctx.meta = {"source_config_path": str(cfg_path)}
            ctx.cfg.runs_dir.return_value = str(base)
            ctx.cfg.handoff_root = str(handoff)
            ctx.cfg.notifications = NotificationConfig(enabled=True)
            ctx.cfg.deployment_file = "deployment.yaml"

            with mock.patch(
                "syndiff_pipeline.template_runner.notifications.post_discord_webhook"
            ) as post:
                send_preview_notification(state, ctx)
                send_preview_notification(state, ctx)
                self.assertEqual(post.call_count, 2)
            with state._conn() as conn:
                n = conn.execute("SELECT COUNT(*) FROM notification_events").fetchone()[0]
            self.assertEqual(n, 0)


class TestNotifier(unittest.TestCase):
    def test_stage_canceled_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            handoff = base

            handoff = base


            db = base / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("r1", "/c", "/t", str(base), [target], ["mapping"])
            cfg_path = base / "config.yaml"
            (base / "deployment.yaml").write_text(
                "discord_webhook_url: https://example.com/hook\n", encoding="utf-8"
            )
            cfg = NotificationConfig(
                enabled=True,
                events=NotificationEvents(stage_canceled=True),
            )
            notifier = Notifier(
                state,
                cfg,
                config_path=cfg_path,
                handoff_root=str(handoff),
                deployment_file="deployment.yaml",
            )
            with mock.patch(
                "syndiff_pipeline.template_runner.notifications.post_discord_webhook"
            ) as post:
                notifier.notify_stage_outcome(
                    "r1",
                    str(base),
                    target_label=target.label(),
                    stage="mapping",
                    outcome="canceled",
                    finished_at="2026-01-01T00:00:00",
                    error_tail="Canceled by user",
                )
                self.assertEqual(post.call_count, 1)
                self.assertIn("stage_canceled", post.call_args[0][1])

    def test_dedup_prevents_second_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            handoff = base

            handoff = base


            db = base / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("r1", "/c", "/t", str(base), [target], ["mapping"])
            cfg_path = base / "config.yaml"
            cfg_path.write_text("data_root: /\n", encoding="utf-8")
            (base / "deployment.yaml").write_text(
                "discord_webhook_url: https://example.com/hook\n", encoding="utf-8"
            )
            cfg = NotificationConfig(
                enabled=True,
                events=NotificationEvents(stage_completed=True),
            )
            notifier = Notifier(
                state,
                cfg,
                config_path=cfg_path,
                handoff_root=str(handoff),
                deployment_file="deployment.yaml",
            )

            with mock.patch(
                "syndiff_pipeline.template_runner.notifications.post_discord_webhook"
            ) as post:
                label = target.label()
                notifier.notify_stage_outcome(
                    "r1",
                    str(base),
                    target_label=label,
                    stage="mapping",
                    outcome="success",
                    finished_at="2026-01-01T00:00:00",
                )
                notifier.notify_stage_outcome(
                    "r1",
                    str(base),
                    target_label=label,
                    stage="mapping",
                    outcome="success",
                    finished_at="2026-01-01T00:00:00",
                )
                self.assertEqual(post.call_count, 1)

    def test_post_discord_webhook_payload(self):
        captured = {}

        def _fake_urlopen(req, timeout=0):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None, read=lambda: b"")

        with mock.patch("urllib.request.urlopen", _fake_urlopen):
            post_discord_webhook("https://example.com/hook", "hello")
        self.assertEqual(captured["body"]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
