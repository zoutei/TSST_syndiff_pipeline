"""Tests for retry and blocked-stage reopen behavior."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner.state import (
    PipelineState,
    STATUS_BLOCKED,
    STATUS_EXTERNAL,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
)
from syndiff_pipeline.template_runner.targets import Target


class TestStageRetry(unittest.TestCase):
    def _target(self) -> Target:
        return Target(
            sector=22,
            camera=3,
            ccd=3,
            target_ra=228.0,
            target_dec=52.0,
            target_name="2020dgc",
        )

    def test_retry_sets_downstream_pending(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            stages = [
                "tess_ffi_download",
                "wcs_grouping",
                "mapping",
                "ps1_download",
                "ps1_process",
                "downsample",
            ]
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, [target], stages)
            state.reset_stage_for_retry("run_a", target.label(), "mapping", reset_downstream=True)

            mapping = state.get_stage_run("run_a", target.label(), "mapping")
            ps1_dl = state.get_stage_run("run_a", target.label(), "ps1_download")
            self.assertEqual(mapping.status, STATUS_PENDING)
            self.assertEqual(ps1_dl.status, STATUS_PENDING)

    def test_list_failed_stage_runs(self):
        t1 = self._target()
        t2 = Target(
            sector=23,
            camera=1,
            ccd=3,
            target_ra=185.0,
            target_dec=5.3,
            target_name="2020ftl",
        )
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            stages = ["mapping", "ps1_process"]
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, [t1, t2], stages)
            state.update_stage_status("run_a", t1.label(), "mapping", STATUS_FAILED, exit_code=1)
            state.update_stage_status("run_a", t2.label(), "ps1_process", STATUS_FAILED, exit_code=1)

            failed = state.list_failed_stage_runs("run_a")
            self.assertEqual(len(failed), 2)

    def test_blocked_downstream_promotes_after_mapping_success(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            stages = [
                "tess_ffi_download",
                "wcs_grouping",
                "mapping",
                "ps1_download",
                "ps1_process",
                "downsample",
            ]
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, [target], stages)
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping", "mapping"):
                state.update_stage_status("run_a", label, stage, STATUS_SUCCESS, exit_code=0)
            state.update_stage_status("run_a", label, "ps1_download", STATUS_BLOCKED)
            state.update_stage_status("run_a", label, "ps1_process", STATUS_BLOCKED)
            state.update_stage_status("run_a", label, "downsample", STATUS_BLOCKED)

            # Skip verification must run before a blocked stage can reach ready.
            state.cache_external_check("run_a", label, "ps1_download", complete=False)
            promoted = state.promote_stages("run_a")

            self.assertEqual(promoted, 1)
            ps1_dl = state.get_stage_run("run_a", label, "ps1_download")
            ps1_proc = state.get_stage_run("run_a", label, "ps1_process")
            down = state.get_stage_run("run_a", label, "downsample")
            self.assertEqual(ps1_dl.status, STATUS_READY)
            self.assertEqual(ps1_proc.status, STATUS_BLOCKED)
            self.assertEqual(down.status, STATUS_BLOCKED)

    def test_bulk_retry_does_not_leave_downsample_ready_without_ps1_process(self):
        target = Target(
            sector=40,
            camera=1,
            ccd=1,
            target_ra=292.646875,
            target_dec=35.776111,
            target_name="2021udg",
        )
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            stages = [
                "tess_ffi_download",
                "wcs_grouping",
                "mapping",
                "ps1_download",
                "ps1_process",
                "downsample",
            ]
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, [target], stages)
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping", "mapping", "ps1_process"):
                state.update_stage_status("run_a", label, stage, STATUS_SUCCESS, exit_code=0)
            state.update_stage_status("run_a", label, "ps1_download", STATUS_FAILED, exit_code=1)
            state.update_stage_status("run_a", label, "downsample", STATUS_FAILED, exit_code=1)

            failed = state.list_failed_stage_runs("run_a")
            for row in failed:
                state.reset_stage_for_retry(
                    "run_a", row.target_label, row.stage, reset_downstream=True
                )

            down = state.get_stage_run("run_a", label, "downsample")
            ps1_proc = state.get_stage_run("run_a", label, "ps1_process")
            ps1_dl = state.get_stage_run("run_a", label, "ps1_download")
            self.assertEqual(ps1_proc.status, STATUS_PENDING)
            self.assertEqual(down.status, STATUS_PENDING)
            self.assertEqual(ps1_dl.status, STATUS_PENDING)
            self.assertFalse(state.deps_satisfied("run_a", label, "downsample"))

    def test_deps_satisfied_accepts_skipped_upstream(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run(
                "run_a",
                "/cfg.yaml",
                "/targets.csv",
                tmp,
                [target],
                ["ps1_process", "downsample"],
            )
            label = target.label()
            state.update_stage_status("run_a", label, "wcs_grouping", STATUS_SKIPPED, exit_code=0)
            state.update_stage_status("run_a", label, "mapping", STATUS_SKIPPED, exit_code=0)
            state.update_stage_status("run_a", label, "ps1_process", STATUS_SUCCESS, exit_code=0)
            self.assertTrue(state.deps_satisfied("run_a", label, "downsample"))

    def test_full_dag_materializes_external_stages(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run(
                "run_a",
                "/cfg.yaml",
                "/targets.csv",
                tmp,
                [target],
                ["downsample"],
            )
            label = target.label()
            mapping = state.get_stage_run("run_a", label, "mapping")
            self.assertEqual(mapping.status, STATUS_EXTERNAL)


if __name__ == "__main__":
    unittest.main()
