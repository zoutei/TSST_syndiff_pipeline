"""Verify run_stage passes downsample progress_path into execute_stage."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template.downsample_progress import PROGRESS_FILENAME


class TestRunStageDownsampleProgressPath(unittest.TestCase):
    def test_execute_stage_receives_progress_path_for_downsample(self):
        captured: dict = {}

        def fake_execute_stage(resolved, stage, force_rerun=False, *, progress_path=None):
            captured["stage"] = stage
            captured["progress_path"] = progress_path
            return None

        fake_ctx = mock.Mock()
        fake_ctx.cfg.runs_dir.return_value = "/runs"
        fake_ctx.targets = [mock.Mock(label=mock.Mock(return_value="t1"))]
        fake_resolved = mock.Mock()

        with mock.patch(
            "syndiff_pipeline.template_runner.run_stage.resolve_run_context",
            return_value=fake_ctx,
        ), mock.patch(
            "syndiff_pipeline.template_runner.run_stage.resolve_config",
            return_value=fake_resolved,
        ), mock.patch(
            "syndiff_pipeline.template_runner.run_stage.stages.execute_stage",
            side_effect=fake_execute_stage,
        ), mock.patch(
            "syndiff_pipeline.template_runner.run_stage.logs.stage_log",
            return_value=mock.Mock(
                __enter__=mock.Mock(return_value=mock.Mock()),
                __exit__=mock.Mock(return_value=False),
            ),
        ), mock.patch(
            "syndiff_pipeline.template_runner.run_stage._write_status",
        ), mock.patch(
            "syndiff_pipeline.template_runner.run_stage.collect_stage_artifacts",
            return_value=(0, 0, []),
        ), mock.patch(
            "syndiff_pipeline.template_runner.run_stage.write_manifest",
        ):
            from syndiff_pipeline.template_runner import run_stage

            rc = run_stage.main(
                [
                    "--run-id",
                    "run_a",
                    "--stage",
                    "downsample",
                    "--run-dir",
                    "/runs/run_a",
                    "--target-label",
                    "t1",
                    "--launch-token",
                    "tok",
                ]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(captured["stage"], "downsample")
        expected = f"/runs/run_a/per_target/t1/{PROGRESS_FILENAME}"
        self.assertEqual(captured["progress_path"], expected)


if __name__ == "__main__":
    unittest.main()
