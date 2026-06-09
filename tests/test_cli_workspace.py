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

from syndiff_pipeline.template_runner import cli
from syndiff_pipeline.template_runner.state import PipelineState
from syndiff_pipeline.template_runner.targets import Target


class TestWorkspaceMonitoring(unittest.TestCase):
    def test_progress_zero_flags_shows_all_active_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir()
            (handoff / "runs").mkdir()
            db = handoff / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            runs_root = str(handoff / "runs")
            state.create_run("run_a", "/c", "/t", runs_root, [target], ["mapping"])
            state.create_run("run_b", "/c", "/t", runs_root, [target], ["mapping"])
            state.set_run_status("run_a", "running")
            state.set_run_status("run_b", "running")

            args = cli.build_parser().parse_args(["progress"])
            with mock.patch.object(
                cli, "_resolve_handoff_from_args", return_value=str(handoff)
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = cli.cmd_progress(args)
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
            db = handoff / "pipeline_state.sqlite"
            state = PipelineState(db)
            target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
            state.create_run("run_old", "/c", "/t", str(runs), [target], ["mapping"])
            state.set_run_status("run_old", "completed")

            args = cli.build_parser().parse_args(["progress"])
            with mock.patch.object(
                cli, "_resolve_handoff_from_args", return_value=str(handoff)
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = cli.cmd_progress(args)
            self.assertEqual(rc, 0)
            self.assertIn("run_old", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
