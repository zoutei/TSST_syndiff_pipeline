"""Tests for downsample Condor executor wiring."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import condor
from syndiff_pipeline.pipeline_spec import SYNDIFF_PIPELINE
from syndiff_pipeline.template_creation.orchestration.runner_config import RunnerConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import parse_stage_params
from syndiff_pipeline.template_creation.orchestration.stages import _condor_resources_for_downsample


class TestDownsampleCondorExecutor(unittest.TestCase):
    def test_parse_stage_params_accepts_downsample_executor(self):
        stages = parse_stage_params(
            {
                "downsample": {
                    "executor": "condor",
                    "condor_request_cpus": 32,
                    "condor_request_memory": 256000,
                }
            }
        )
        self.assertEqual(stages.downsample.executor, "condor")
        self.assertEqual(stages.downsample.condor_request_cpus, 32)
        self.assertEqual(stages.downsample.condor_request_memory, 256000)

    def test_parse_stage_params_default_downsample_executor_local(self):
        stages = parse_stage_params({})
        self.assertEqual(stages.downsample.executor, "local")

    def test_parse_stage_params_rejects_legacy_condor_keys(self):
        with self.assertRaises(ValueError) as ctx:
            parse_stage_params(
                {
                    "mapping": {
                        "condor_requirements": "LoadAvg < 10",
                    }
                }
            )
        self.assertIn("condor_requirements", str(ctx.exception))

    def test_resolve_executor_downsample_condor(self):
        cfg = RunnerConfig(
            stages=parse_stage_params({"downsample": {"executor": "condor"}}),
            workspace_root="/tmp/ws",
            data_root="/tmp/data",
        )
        spec = SYNDIFF_PIPELINE.require("downsample")
        self.assertEqual(spec.resolve_executor(cfg), "condor")

    def test_condor_resources_for_downsample(self):
        cfg = RunnerConfig(
            stages=parse_stage_params(
                {
                    "downsample": {
                        "executor": "condor",
                        "condor_request_cpus": 64,
                        "condor_request_memory": 500000,
                        "condor_request_disk": 10000,
                        "host_stats_min_mem_mb": 500000,
                    }
                }
            ),
            workspace_root="/tmp/ws",
            data_root="/tmp/data",
        )
        req = _condor_resources_for_downsample(cfg)
        self.assertEqual(req.request_cpus, 64)
        self.assertEqual(req.request_memory_mb, 500000)
        self.assertEqual(req.request_disk_kb, 10000 * 1024)
        self.assertEqual(req.host_stats_min_mem_mb, 500000)
        self.assertIsNone(req.requirements)

    def test_write_submit_file_includes_request_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            submit = Path(tmpdir) / "job.submit"
            artifacts = {
                "stdout": Path(tmpdir) / "out",
                "stderr": Path(tmpdir) / "err",
                "log": Path(tmpdir) / "log",
            }
            condor.write_submit_file(
                submit,
                ["python", "-m", "syndiff_pipeline.common.orchestration.run_stage"],
                artifacts,
                condor.CondorResourceRequest(
                    request_cpus=8,
                    request_memory_mb=192000,
                    request_disk_kb=10000 * 1024,
                ),
            )
            text = submit.read_text()
            self.assertIn("request_disk = 10240000", text)
            self.assertIn("request_cpus = 8", text)

    def test_write_submit_file_omits_disk_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            submit = Path(tmpdir) / "job.submit"
            artifacts = {
                "stdout": Path(tmpdir) / "out",
                "stderr": Path(tmpdir) / "err",
                "log": Path(tmpdir) / "log",
            }
            condor.write_submit_file(
                submit,
                ["python", "-c", "pass"],
                artifacts,
                condor.CondorResourceRequest(request_disk_kb=None),
            )
            self.assertNotIn("request_disk", submit.read_text())

    def test_launch_stage_submits_downsample_to_condor(self):
        from syndiff_pipeline.common.orchestration.launcher import launch_stage

        cfg = RunnerConfig(
            stages=parse_stage_params({"downsample": {"executor": "condor"}}),
            workspace_root="/tmp/ws",
            data_root="/tmp/data",
        )
        with mock.patch(
            "syndiff_pipeline.common.orchestration.launcher.condor.submit_job",
            return_value=(99, 1.0),
        ) as submit_job:
            desc = launch_stage(
                ["python", "-m", "syndiff_pipeline.common.orchestration.run_stage"],
                cfg=cfg,
                stage="downsample",
                runs_root="/runs",
                run_id="run_a",
                target_label="s0020_c3_k3_2020ut",
                launch_token="tok",
            )
        self.assertEqual(desc.executor, "condor")
        self.assertEqual(desc.native_id, 99)
        submit_job.assert_called_once()
        resources = submit_job.call_args.kwargs["resources"]
        self.assertIsInstance(resources, condor.CondorResourceRequest)


if __name__ == "__main__":
    unittest.main()
