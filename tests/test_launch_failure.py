"""Tests for launch failure requeue cap and backoff."""
from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.scheduler import _try_launch_ready_row
from syndiff_pipeline.common.orchestration.state import (
    PipelineState,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    _utc_now,
)
from syndiff_pipeline.common.orchestration.targets import Target
from tests.test_daemon_behavior import _minimal_run_setup


def _target() -> Target:
    return Target(
        sector=40,
        camera=1,
        ccd=1,
        target_ra=292.646875,
        target_dec=35.776111,
        target_name="2021udg",
    )


def _set_stage_fields(
    state: PipelineState,
    run_id: str,
    label: str,
    stage: str,
    **fields,
) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [run_id, label, stage]
    with state._conn() as conn:
        conn.execute(
            f"UPDATE stage_runs SET {sets} WHERE run_id = ? AND target_label = ? AND stage = ?",
            values,
        )


def _launch_ready_row(state, ctx, run_id, label, stage, *, runs_root):
    row = state.get_stage_run(run_id, label, stage)
    return _try_launch_ready_row(
        state,
        run_id,
        ctx,
        row,
        pool_label="network",
        force_rerun=False,
        active_stages=[stage],
        targets_by_label={label: _target()},
        runs_root=str(runs_root),
    )


def _prepare_ps1_download_launch(state, run_id, label):
    """Satisfy ps1_download upstream deps so launch can proceed."""
    state.update_stage_status(run_id, label, "mapping", STATUS_SUCCESS, exit_code=0)
    state.update_stage_status(run_id, label, "ps1_download", STATUS_READY)


class TestLaunchFailureRequeue(unittest.TestCase):
    def test_launch_failure_sets_backoff(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_download"]
            )
            label = target.label()
            _prepare_ps1_download_launch(state, run_id, label)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=RuntimeError("spawn failed"),
            ):
                launched = _launch_ready_row(
                    state, ctx, run_id, label, "ps1_download", runs_root=runs_root
                )

            self.assertFalse(launched)
            row = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(row.status, STATUS_READY)
            self.assertEqual(row.attempts, 1)
            self.assertIsNotNone(row.not_before)
            self.assertGreater(row.not_before, _utc_now())

    def test_launch_failure_fails_after_max_attempts(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_download"]
            )
            label = target.label()
            _prepare_ps1_download_launch(state, run_id, label)
            _set_stage_fields(state, run_id, label, "ps1_download", attempts=2)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=RuntimeError("spawn failed"),
            ):
                launched = _launch_ready_row(
                    state, ctx, run_id, label, "ps1_download", runs_root=runs_root
                )

            self.assertFalse(launched)
            row = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(row.status, STATUS_FAILED)
            self.assertEqual(row.attempts, 3)
            self.assertIn("gave up after 3 attempts", row.error_tail or "")

    def test_launch_failure_does_not_double_count_attempts(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_download"]
            )
            label = target.label()
            _prepare_ps1_download_launch(state, run_id, label)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=RuntimeError("spawn failed"),
            ):
                _launch_ready_row(
                    state, ctx, run_id, label, "ps1_download", runs_root=runs_root
                )

            row = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(row.attempts, 1)
            self.assertEqual(row.status, STATUS_READY)

            past = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
            _set_stage_fields(state, run_id, label, "ps1_download", not_before=past)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=RuntimeError("spawn failed"),
            ):
                _launch_ready_row(
                    state, ctx, run_id, label, "ps1_download", runs_root=runs_root
                )

            row = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(row.attempts, 2)
            self.assertEqual(row.status, STATUS_READY)

    def test_launch_failure_leaves_running_only_during_claim(self):
        """Failed launch must not leave the stage stuck in running."""
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_download"]
            )
            label = target.label()
            _prepare_ps1_download_launch(state, run_id, label)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=RuntimeError("spawn failed"),
            ):
                _launch_ready_row(
                    state, ctx, run_id, label, "ps1_download", runs_root=runs_root
                )

            row = state.get_stage_run(run_id, label, "ps1_download")
            self.assertNotEqual(row.status, STATUS_RUNNING)


if __name__ == "__main__":
    unittest.main()
