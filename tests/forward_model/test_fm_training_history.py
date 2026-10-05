# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Unit tests for training_history progress helpers."""

from __future__ import annotations

import unittest

from syndiff_pipeline.forward_model.training_history import overall_progress, stage_budgets


class StageBudgetsTests(unittest.TestCase):
    def test_full_run_from_stage_one(self):
        controller = {"steps_per_stage": "200,4000,5000", "first_stage": 1}
        self.assertEqual(stage_budgets(controller), [(1, 200), (2, 4000), (3, 5000)])

    def test_bootstrap_stage_three_uses_third_slot(self):
        controller = {"steps_per_stage": "0,0,5000", "first_stage": 3}
        self.assertEqual(stage_budgets(controller), [(1, 0), (2, 0), (3, 5000)])

    def test_overall_progress_bootstrap_stage_three(self):
        budgets = [(1, 0), (2, 0), (3, 5000)]
        frac, done, total = overall_progress(3, 1693, budgets)
        self.assertEqual(total, 5000)
        self.assertEqual(done, 1694)
        self.assertAlmostEqual(frac, 1694 / 5000)


if __name__ == "__main__":
    unittest.main()
