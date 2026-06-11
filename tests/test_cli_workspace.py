"""Tests for workspace-scoped CLI monitoring."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import cli as orch_cli
from syndiff_pipeline.cli import parse_execution_argv
from syndiff_pipeline.common.orchestration.state import PipelineState
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.orchestration.workspace import state_db_path


class TestWorkspaceMonitoring(unittest.TestCase):
    def test_progress_zero_flags_shows_all_active_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir()
            (handoff / "runs").mkdir()
            db = state_db_path(handoff)
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            runs_root = str(handoff / "runs")
            state.create_run("run_a", "/c", "/t", runs_root, [target], ["mapping"])
            state.create_run("run_b", "/c", "/t", runs_root, [target], ["mapping"])
            state.set_run_status("run_a", "running")
            state.set_run_status("run_b", "running")

            args = orch_cli.build_parser().parse_args(["progress"])
            with mock.patch.object(
                orch_cli, "_resolve_handoff_from_args", return_value=str(handoff)
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = orch_cli.cmd_progress(args)
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("run_a", out)
            self.assertIn("run_b", out)

    def test_progress_falls_back_to_latest_when_none_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            runs = handoff / "runs"
            runs.mkdir(parents=True)
            (runs / "latest").symlink_to("run_old")
            db = state_db_path(handoff)
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("run_old", "/c", "/t", str(runs), [target], ["mapping"])
            state.set_run_status("run_old", "completed")

            args = orch_cli.build_parser().parse_args(["progress"])
            with mock.patch.object(
                orch_cli, "_resolve_handoff_from_args", return_value=str(handoff)
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = orch_cli.cmd_progress(args)
            self.assertEqual(rc, 0)
            self.assertIn("run_old", buf.getvalue())

    def test_main_entry_routes_monitoring_verbs(self):
        from syndiff_pipeline import cli as entry_cli

        with mock.patch.object(orch_cli, "main", return_value=0) as mocked:
            rc = entry_cli.main(["progress"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once_with(["progress"])


class TestSubmitRunIdPolicy(unittest.TestCase):
    def test_submit_rejects_duplicate_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir()
            db = state_db_path(handoff)
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run(
                "batch_a",
                "/c",
                "/t",
                str(handoff / "runs"),
                [target],
                ["wcs_grouping"],
            )

            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        f"workspace_root: {handoff}",
                        f"data_root: {handoff / 'data'}",
                        "deployment_file: deployment.yaml",
                    ]
                ),
                encoding="utf-8",
            )
            (Path(tmp) / "deployment.yaml").write_text(
                f"workspace_root: {handoff}\ndata_root: {handoff / 'data'}\n",
                encoding="utf-8",
            )
            targets_path = Path(tmp) / "targets.csv"
            targets_path.write_text(
                "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
                "22,3,3,228.0,52.0,2020dgc,true\n",
                encoding="utf-8",
            )

            _, _, args = parse_execution_argv(
                [
                    "template",
                    "submit",
                    "--config",
                    str(cfg_path),
                    "--targets",
                    str(targets_path),
                    "--run-id",
                    "batch_a",
                    "--stages",
                    "wcs_grouping",
                ]
            )
            with mock.patch.object(
                orch_cli, "load_runner_config", return_value=mock.Mock(
                    workspace_root=str(handoff),
                    runs_dir=lambda: str(handoff / "runs"),
                    state_db_path=str(db),
                    deployment_file="deployment.yaml",
                    stages=mock.Mock(
                        ps1_process=mock.Mock(ps1_source="zarr"),
                    ),
                    notifications=mock.Mock(enabled=False),
                )
            ), mock.patch.object(orch_cli, "ensure_daemon_running"), mock.patch.object(
                orch_cli, "_ensure_discord_bot", return_value=None
            ), mock.patch.object(
                orch_cli, "record_deployment_path"
            ):
                with self.assertRaises(SystemExit) as ctx:
                    orch_cli.cmd_submit(args)
            self.assertIn("already exists", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
