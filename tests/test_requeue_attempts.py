"""Tests for stage requeue attempt limits and backoff."""
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

from syndiff_pipeline.common.orchestration.scheduler import reconcile_running_stages
from syndiff_pipeline.common.orchestration.state import (
    PipelineState,
    STATUS_BLOCKED,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_RUNNING,
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


class TestClaimIncrementsAttempts(unittest.TestCase):
    def test_claim_ready_increments_attempts(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, _ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [_target()], active_stages=["mapping"]
            )
            label = target.label()
            state.update_stage_status(run_id, label, "mapping", STATUS_READY)

            self.assertTrue(state.claim_ready(run_id, label, "mapping", state.new_launch_token()))
            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.attempts, 1)

            state.requeue_to_ready(run_id, label, "mapping")
            state.update_stage_status(run_id, label, "mapping", STATUS_READY)
            self.assertTrue(state.claim_ready(run_id, label, "mapping", state.new_launch_token()))
            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.attempts, 2)


class TestRequeueBackoff(unittest.TestCase):
    def test_requeue_sets_not_before_and_fetch_respects_it(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_download"]
            )
            label = target.label()
            token = state.new_launch_token()
            state.update_stage_status(run_id, label, "ps1_download", STATUS_READY)
            state.try_atomic_claim(
                run_id,
                label,
                "ps1_download",
                launch_token=token,
                executor="local",
                native_id=999999,
                log_path=str(tmp_path / "x.log"),
            )

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.daemon.is_process_alive",
                return_value=False,
            ):
                counts = reconcile_running_stages(state, run_id, ctx)

            row = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(counts["requeued"], 1)
            self.assertEqual(row.status, STATUS_READY)
            self.assertIsNotNone(row.not_before)
            self.assertGreater(row.not_before, _utc_now())

            batch = state.fetch_ready_batch(run_id, "network", limit=10)
            self.assertEqual(batch, [])
            unpooled = state.fetch_ready_unpooled(run_id)
            self.assertEqual(unpooled, [])

            past = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
            _set_stage_fields(state, run_id, label, "ps1_download", not_before=past)
            batch = state.fetch_ready_batch(run_id, "network", limit=10)
            self.assertEqual(len(batch), 1)
            self.assertEqual(batch[0].stage, "ps1_download")


class TestMaxAttemptsGiveUp(unittest.TestCase):
    def test_reconcile_fails_after_max_attempts(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping", "ps1_download"]
            )
            label = target.label()
            token = state.new_launch_token()
            state.update_stage_status(run_id, label, "mapping", STATUS_READY)
            state.try_atomic_claim(
                run_id,
                label,
                "mapping",
                launch_token=token,
                executor="local",
                native_id=999999,
                log_path=str(tmp_path / "x.log"),
            )
            _set_stage_fields(state, run_id, label, "mapping", attempts=3)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.daemon.is_process_alive",
                return_value=False,
            ):
                counts = reconcile_running_stages(state, run_id, ctx)

            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(counts["requeued"], 0)
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(row.status, STATUS_FAILED)
            self.assertIn("gave up after 3 attempts", row.error_tail or "")
            self.assertIsNone(row.native_id)
            self.assertIsNone(row.launch_token)

            ps1_row = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(ps1_row.status, STATUS_BLOCKED)


class TestHumanRetryResetsAttempts(unittest.TestCase):
    def test_reset_stage_for_retry_clears_attempts_and_not_before(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, _ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            _set_stage_fields(
                state,
                run_id,
                label,
                "mapping",
                status=STATUS_FAILED,
                attempts=3,
                not_before=future,
            )

            state.reset_stage_for_retry(run_id, label, "mapping", reset_downstream=False)
            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.attempts, 0)
            self.assertIsNone(row.not_before)


if __name__ == "__main__":
    unittest.main()
