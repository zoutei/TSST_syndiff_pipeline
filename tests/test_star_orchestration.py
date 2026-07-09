"""Tests for star orchestration stage spec."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.pipeline_spec import get_syndiff_pipeline
from syndiff_pipeline.star.orchestration.stages import STAR_STAGE, _verify_star


class TestStarOrchestration(unittest.TestCase):
    def test_star_stage_registered_in_pipeline(self):
        pipeline = get_syndiff_pipeline()
        self.assertIn("star", pipeline.stage_names)
        self.assertEqual(pipeline.get("star").pool, "star")

    def test_star_stage_has_no_orchestrator_deps(self):
        self.assertEqual(STAR_STAGE.deps, ())

    def test_verify_star_false_when_manifest_missing(self):
        ctx = SimpleNamespace(
            run_id="r1",
            runs_root="/tmp/runs",
            target_label="s0020_c3_k2_s20_astrometry",
            target=SimpleNamespace(
                sector=20,
                camera=3,
                ccd=2,
                target_name="s20_astrometry",
            ),
            runner_cfg=SimpleNamespace(star_config_path="/tmp/star_config.yaml"),
            meta={"source_star_config_path": "/tmp/star_config.yaml"},
        )
        with mock.patch(
            "syndiff_pipeline.star.orchestration.stages._resolve_star_run",
            side_effect=FileNotFoundError("missing"),
        ):
            self.assertFalse(_verify_star(ctx))


if __name__ == "__main__":
    unittest.main()
