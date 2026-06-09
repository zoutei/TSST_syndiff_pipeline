"""Tests for PS1 stream mode (on-the-fly download) and skip semantics."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner.runner_config import RunnerConfig, parse_stage_params, resolve_config
from syndiff_pipeline.template_runner.stage_params import Ps1ProcessStageParams
from syndiff_pipeline.template_runner.state import (
    SKIP_REASON_ARTIFACTS,
    SKIP_REASON_STREAM,
    STATUS_PENDING,
    STATUS_SKIPPED,
    PipelineState,
    STAGE_NAMES,
    effective_stage_deps,
)
from syndiff_pipeline.template_runner.targets import Target
from syndiff_pipeline.template_runner.run_report import _format_stage_status_short


class TestEffectiveStageDeps(unittest.TestCase):
    def test_stream_mode_skips_ps1_download_dep(self):
        stages = parse_stage_params({"ps1_process": {"ps1_source": "stream"}})
        self.assertEqual(effective_stage_deps("ps1_process", stages), ["mapping"])

    def test_zarr_mode_keeps_ps1_download_dep(self):
        stages = parse_stage_params({})
        self.assertEqual(effective_stage_deps("ps1_process", stages), ["ps1_download"])


class TestStreamSkipAtCreate(unittest.TestCase):
    def test_apply_ps1_stream_download_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            state = PipelineState(db)
            cfg = RunnerConfig(
                data_root="/data",
                ffi_dir="/data",
                handoff_root=str(Path(tmp) / "handoffs"),
                runs_root=str(Path(tmp) / "runs"),
                state_db_path=str(db),
                skycell_wcs_csv="/wcs.csv",
                stages=parse_stage_params({"ps1_process": {"ps1_source": "stream"}}),
            )
            target = Target(80, 4, 2, 274.9, 66.0, "2024pvw", True)
            state.create_run(
                "batch_test",
                str(Path(tmp) / "config.yaml"),
                str(Path(tmp) / "targets.csv"),
                cfg.runs_root,
                [target],
                ["mapping", "ps1_process", "downsample"],
            )
            n = state.apply_ps1_stream_download_skips("batch_test", [target], cfg)
            self.assertEqual(n, 1)
            row = state.get_stage_run("batch_test", target.label(), "ps1_download")
            self.assertEqual(row.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason("batch_test", target.label(), "ps1_download"),
                SKIP_REASON_STREAM,
            )
            state.update_stage_status(
                "batch_test", target.label(), "mapping", "success", exit_code=0
            )
            self.assertTrue(
                state.deps_satisfied(
                    "batch_test",
                    target.label(),
                    "ps1_process",
                    stages=resolve_config(target, cfg).stages,
                )
            )


class TestStatusDisplay(unittest.TestCase):
    def test_stream_skip_shows_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            state = PipelineState(db)
            target = Target(80, 4, 2, 274.9, 66.0, "2024pvw", True)
            state.create_run(
                "r1",
                "cfg",
                "targets",
                str(Path(tmp) / "runs"),
                [target],
                list(STAGE_NAMES),
            )
            state.mark_skipped("r1", target.label(), "ps1_download")
            state.cache_skip_reason("r1", target.label(), "ps1_download", SKIP_REASON_STREAM)
            row = state.get_stage_run("r1", target.label(), "ps1_download")
            self.assertEqual(_format_stage_status_short(state, "r1", row), "ps1_dl:n/a")

    def test_artifact_skip_shows_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            state = PipelineState(db)
            target = Target(80, 4, 2, 274.9, 66.0, "2024pvw", True)
            state.create_run(
                "r1",
                "cfg",
                "targets",
                str(Path(tmp) / "runs"),
                [target],
                list(STAGE_NAMES),
            )
            state.mark_skipped("r1", target.label(), "ps1_download")
            state.cache_skip_reason("r1", target.label(), "ps1_download", SKIP_REASON_ARTIFACTS)
            row = state.get_stage_run("r1", target.label(), "ps1_download")
            self.assertEqual(_format_stage_status_short(state, "r1", row), "ps1_dl:skip")


class TestStageParams(unittest.TestCase):
    def test_invalid_ps1_source_raises(self):
        with self.assertRaises(ValueError):
            parse_stage_params({"ps1_process": {"ps1_source": "ftp"}})

    def test_defaults(self):
        stages = parse_stage_params({})
        self.assertEqual(stages.ps1_process.ps1_source, "zarr")
        self.assertEqual(stages.ps1_process.num_ingest_workers, 16)


if __name__ == "__main__":
    unittest.main()
