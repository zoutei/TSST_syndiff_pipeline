"""Tests for PS1 stream mode (on-the-fly download) and skip semantics."""
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

from syndiff_pipeline.common.orchestration.run_setup import apply_post_create_run_setup
from syndiff_pipeline.common.orchestration.scheduler import run_scheduler
from syndiff_pipeline.template_creation.orchestration.runner_config import RunnerConfig, parse_stage_params, resolve_config
from syndiff_pipeline.template_creation.orchestration.stage_params import Ps1ProcessStageParams
from syndiff_pipeline.common.orchestration.state import (
    SKIP_REASON_ARTIFACTS,
    SKIP_REASON_STREAM,
    STATUS_PENDING,
    STATUS_SKIPPED,
    PipelineState,
    STAGE_NAMES,
    effective_stage_deps,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.run_report import _format_stage_status_short


class TestEffectiveStageDeps(unittest.TestCase):
    def test_stream_mode_skips_ps1_download_dep(self):
        stages = parse_stage_params({"ps1_process": {"ps1_source": "stream"}})
        self.assertEqual(effective_stage_deps("ps1_process", stages), ["mapping"])

    def test_zarr_mode_keeps_ps1_download_dep(self):
        stages = parse_stage_params({})
        self.assertEqual(effective_stage_deps("ps1_process", stages), ["ps1_download"])


def _write_stream_run_dir(tmp: Path, *, run_id: str, db_path: Path) -> Path:
    handoff = tmp / "handoff"
    data = tmp / "data"
    runs_root = handoff / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "per_target").mkdir()
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"data_root: {data}",
                f"workspace_root: {handoff}",
                f"runs_root: {runs_root}",
                f"state_db_path: {db_path}",
                "stages:",
                "  mapping: {}",
                "  ps1_process:",
                "    ps1_source: stream",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "targets.csv").write_text(
        "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
        "80,4,2,274.9,66.0,2024pvw,true\n",
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps({"run_id": run_id}),
        encoding="utf-8",
    )
    return run_dir


class TestStreamSkipAtCreate(unittest.TestCase):
    def test_apply_post_create_run_setup_stream_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = tmp_path / "state.sqlite"
            cfg = RunnerConfig(
                data_root="/data",
                ffi_dir="/data",
                workspace_root=str(tmp_path / "handoffs"),
                runs_root=str(tmp_path / "runs"),
                skycell_wcs_csv="/wcs.csv",
                stages=parse_stage_params({"ps1_process": {"ps1_source": "stream"}}),
            )
            state = PipelineState(db)
            target = Target(80, 4, 2, 274.9, 66.0, "2024pvw", True)
            state.create_run(
                "batch_test",
                str(tmp_path / "config.yaml"),
                str(tmp_path / "targets.csv"),
                cfg.runs_root,
                [target],
                ["mapping", "ps1_process", "downsample"],
            )
            result = apply_post_create_run_setup(
                state, "batch_test", [target], cfg, ["mapping", "ps1_process", "downsample"]
            )
            self.assertEqual(result.stream_skipped, 1)
            row = state.get_stage_run("batch_test", target.label(), "ps1_download")
            self.assertEqual(row.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason("batch_test", target.label(), "ps1_download"),
                SKIP_REASON_STREAM,
            )

    def test_apply_ps1_stream_download_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            state = PipelineState(db)
            cfg = RunnerConfig(
                data_root="/data",
                ffi_dir="/data",
                workspace_root=str(Path(tmp) / "handoffs"),
                runs_root=str(Path(tmp) / "runs"),
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


class TestRunSchedulerStreamSkips(unittest.TestCase):
    @mock.patch("syndiff_pipeline.common.orchestration.scheduler._tick_run")
    def test_run_scheduler_applies_stream_skips_after_create(self, mock_tick):
        def finish_run(state, run_id, ctx):
            state.set_run_status(run_id, "success")

        mock_tick.side_effect = finish_run

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = tmp_path / "state.sqlite"
            run_dir = _write_stream_run_dir(tmp_path, run_id="stream_run", db_path=db)
            rc = run_scheduler(
                "stream_run",
                str(run_dir),
                stages_arg="mapping,ps1_process,downsample",
            )
            self.assertEqual(rc, 0)
            state = PipelineState(db)
            row = state.get_stage_run("stream_run", "s0080_c4_k2_2024pvw", "ps1_download")
            self.assertEqual(row.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason("stream_run", "s0080_c4_k2_2024pvw", "ps1_download"),
                SKIP_REASON_STREAM,
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
