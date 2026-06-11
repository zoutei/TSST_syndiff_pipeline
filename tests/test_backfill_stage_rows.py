"""Tests for missing stage_runs backfill."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.run_setup import apply_post_create_run_setup
from syndiff_pipeline.common.orchestration.state import (
    PipelineState,
    SKIP_REASON_NOT_SELECTED,
    STATUS_EXTERNAL,
    STATUS_SKIPPED,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.run_report import format_status_grid
from tests.test_daemon_behavior import _minimal_run_setup


class TestBackfillStageRows(unittest.TestCase):
    def test_backfill_inserts_missing_diff_and_shows_na_in_grid(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping", "downsample"]
            )
            label = target.label()
            with state._conn() as conn:
                conn.execute(
                    "DELETE FROM stage_runs WHERE run_id = ? AND stage = ?",
                    (run_id, "diff"),
                )

            self.assertIsNone(state.get_stage_run(run_id, label, "diff"))
            inserted = state.backfill_missing_stage_rows(run_id)
            self.assertEqual(inserted, 1)
            diff = state.get_stage_run(run_id, label, "diff")
            self.assertIsNotNone(diff)
            self.assertEqual(diff.status, STATUS_EXTERNAL)

            inserted_again = state.backfill_missing_stage_rows(run_id)
            self.assertEqual(inserted_again, 0)

            apply_post_create_run_setup(
                state, run_id, ctx.targets, ctx.cfg, state.get_active_stages(run_id)
            )
            diff = state.get_stage_run(run_id, label, "diff")
            self.assertEqual(diff.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "diff"),
                SKIP_REASON_NOT_SELECTED,
            )

            lines = format_status_grid(state, run_id)
            self.assertEqual(len(lines), 1)
            self.assertIn("diff:n/a", lines[0])

    def test_backfill_no_targets_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            self.assertEqual(state.backfill_missing_stage_rows("missing_run"), 0)


if __name__ == "__main__":
    unittest.main()
