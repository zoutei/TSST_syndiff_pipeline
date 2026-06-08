"""Tests for Condor poll grace and RA normalization helpers."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template.pancakes import (
    moc_ra_shift_degrees,
    normalize_ra_degrees,
    shift_polygon_ras_for_moc,
    shift_ras_for_moc,
)
from syndiff_pipeline.template_runner import condor, logs
from syndiff_pipeline.template_runner.run_context import resolve_run_context
from syndiff_pipeline.template_runner.scheduler import reconcile_running_stages
from syndiff_pipeline.template_runner.state import (
    PipelineState,
    STAGE_DEPS,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_RUNNING,
)
from syndiff_pipeline.template_runner.targets import Target


class TestCondorPollGrace(unittest.TestCase):
    def test_poll_returns_none_within_grace_when_missing(self):
        cluster_id = 999_001
        condor._submission_times[cluster_id] = time.time()
        with unittest.mock.patch.object(condor, "_query_queue", return_value=(None, None)):
            with unittest.mock.patch.object(condor, "_query_history", return_value=(None, None)):
                self.assertIsNone(condor.poll_cluster(cluster_id))

    def test_poll_returns_failure_after_grace_when_missing(self):
        cluster_id = 999_002
        condor._submission_times[cluster_id] = time.time() - condor.poll_grace_seconds() - 1.0
        with unittest.mock.patch.object(condor, "_query_queue", return_value=(None, None)):
            with unittest.mock.patch.object(condor, "_query_history", return_value=(None, None)):
                self.assertEqual(condor.poll_cluster(cluster_id), 1)

    def test_poll_removed_with_exit_zero_is_canceled_not_success(self):
        cluster_id = 999_003
        with unittest.mock.patch.object(
            condor, "_query_queue", return_value=(None, None)
        ), unittest.mock.patch.object(
            condor, "_query_history", return_value=(condor._JOB_REMOVED, 0)
        ):
            self.assertEqual(condor.poll_cluster(cluster_id), 143)


def _minimal_condor_run(tmp: Path, target: Target) -> tuple[PipelineState, str]:
    state_db = tmp / "state.sqlite"
    runs_root = tmp / "runs"
    run_id = "run_a"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "per_target").mkdir()
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data_root: /data",
                f"handoff_root: {tmp}",
                f"runs_root: {runs_root}",
                f"state_db_path: {state_db}",
                "skycell_wcs_csv: skycells.csv",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "targets.csv").write_text(
        "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
        f"{target.sector},{target.camera},{target.ccd},1,1,{target.target_name},true\n",
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    state = PipelineState(str(state_db))
    state.create_run(
        run_id,
        str(cfg_path),
        str(run_dir / "targets.csv"),
        str(runs_root),
        [target],
        ["ps1_process"],
    )
    return state, str(run_dir)


class TestCondorGraceAcrossRestart(unittest.TestCase):
    """Reconcile must use the DB-persisted wall-clock submit_epoch for the poll
    grace, because the in-process ``_submission_times`` map is empty after a
    daemon restart."""

    def _target(self) -> Target:
        return Target(
            sector=40,
            camera=1,
            ccd=1,
            target_ra=292.6,
            target_dec=35.7,
            target_name="2021udg",
        )

    def _claimed_condor_run(self, tmp: Path, cluster_id: int, submit_epoch: float):
        target = self._target()
        state, run_dir = _minimal_condor_run(tmp, target)
        label = target.label()
        # Simulate a restart: the in-process submission map has no record.
        condor._submission_times.pop(cluster_id, None)
        state.update_stage_status("run_a", label, "ps1_process", STATUS_READY)
        state.try_atomic_claim(
            "run_a",
            label,
            "ps1_process",
            launch_token=state.new_launch_token(),
            executor="condor",
            native_id=cluster_id,
            log_path=str(logs.target_log_path(str(tmp / "runs"), "run_a", label, "ps1_process")),
            submit_epoch=submit_epoch,
        )
        ctx = resolve_run_context(run_dir=run_dir)
        return state, ctx, label

    def test_briefly_missing_cluster_within_grace_not_failed(self):
        cluster_id = 888_001
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, label = self._claimed_condor_run(
                tmp_path, cluster_id, submit_epoch=time.time()
            )
            with unittest.mock.patch.object(
                condor, "_query_queue", return_value=(None, None)
            ), unittest.mock.patch.object(
                condor, "_query_history", return_value=(None, None)
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(counts["failed"], 0)
            self.assertEqual(row.status, STATUS_RUNNING)

    def test_missing_cluster_past_grace_is_failed(self):
        cluster_id = 888_002
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            past = time.time() - condor.poll_grace_seconds() - 1.0
            state, ctx, label = self._claimed_condor_run(
                tmp_path, cluster_id, submit_epoch=past
            )
            with unittest.mock.patch.object(
                condor, "_query_queue", return_value=(None, None)
            ), unittest.mock.patch.object(
                condor, "_query_history", return_value=(None, None)
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(row.status, STATUS_FAILED)


class TestStageDeps(unittest.TestCase):
    def test_downsample_requires_mapping(self):
        self.assertEqual(STAGE_DEPS["downsample"], ["mapping", "ps1_process"])


class TestRaNormalization(unittest.TestCase):
    def test_normalize_ra_degrees_wraps(self):
        ra = normalize_ra_degrees(np.array([-10.0, 370.0, 358.0]))
        np.testing.assert_allclose(ra, [350.0, 10.0, 358.0])

    def test_shift_polygon_ras_for_moc_spans_zero(self):
        vertices = np.array(
            [
                [[359.0, 0.0], [1.0, 0.0], [1.0, 1.0], [359.0, 1.0]],
            ],
            dtype=np.float64,
        )
        shift = moc_ra_shift_degrees(358.0)
        out = shift_polygon_ras_for_moc(vertices, shift)
        self.assertTrue(np.all(out[:, :, 0] >= 0.0))
        self.assertTrue(np.all(out[:, :, 0] < 360.0))
        self.assertLess(out[0, :, 0].max() - out[0, :, 0].min(), 180.0)

    def test_shift_ras_for_moc_near_zero(self):
        shift = moc_ra_shift_degrees(358.0)
        ra = shift_ras_for_moc(np.array([359.0, 0.5, 1.0]), shift)
        self.assertTrue(np.all(ra >= 0.0))
        self.assertTrue(np.all(ra < 360.0))


if __name__ == "__main__":
    unittest.main()
