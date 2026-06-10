"""Tests for force-rerun scheduler bookkeeping."""
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
    STATUS_PENDING,
    STATUS_SUCCESS,
)
from syndiff_pipeline.template_runner.targets import Target
from syndiff_pipeline.template_runner.runner_config import resolve_config, RunnerConfig
from syndiff_pipeline.template_runner.verify import clear_ps1_process_artifacts


class TestForceRerun(unittest.TestCase):
    def test_reset_stages_for_force_rerun(self):
        target = Target(
            sector=23,
            camera=1,
            ccd=3,
            target_ra=185.0,
            target_dec=5.3,
            target_name="2020ftl",
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            state = PipelineState(str(db))
            state.create_run(
                "run_a",
                "/cfg.yaml",
                "/targets.csv",
                tmp,
                [target],
                ["mapping"],
            )
            state.update_stage_status(
                "run_a", target.label(), "mapping", STATUS_SUCCESS, exit_code=0
            )
            state.reset_stages_for_force_rerun("run_a", [target.label()], ["mapping"])
            row = state.get_stage_run("run_a", target.label(), "mapping")
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.status, STATUS_PENDING)
            self.assertIsNone(row.exit_code)

    def test_apply_force_rerun_updates_selected_stages(self):
        target = Target(
            sector=23,
            camera=1,
            ccd=3,
            target_ra=185.0,
            target_dec=5.3,
            target_name="2020ftl",
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            state = PipelineState(str(db))
            state.create_run(
                "run_a",
                "/cfg.yaml",
                "/targets.csv",
                tmp,
                [target],
                ["wcs_grouping"],
            )
            label = target.label()
            state.apply_not_selected_skips(
                "run_a",
                [target],
                RunnerConfig(data_root=str(Path(tmp) / "data")),
            )
            self.assertEqual(
                state.get_skip_reason("run_a", label, "downsample"),
                "not_selected",
            )
            state.apply_force_rerun(
                "run_a",
                [label],
                ["wcs_grouping", "downsample"],
            )
            self.assertEqual(
                state.get_active_stages("run_a"),
                ["wcs_grouping", "downsample"],
            )
            down = state.get_stage_run("run_a", label, "downsample")
            self.assertIsNotNone(down)
            assert down is not None
            self.assertEqual(down.status, STATUS_PENDING)

    def test_clear_ps1_process_artifacts(self):
        target = Target(
            sector=15,
            camera=1,
            ccd=4,
            target_ra=100.0,
            target_dec=20.0,
            target_name="2019pdx",
        )
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            convolved = (
                data_root
                / "convolved_results"
                / "sector_0015_camera_1_ccd_4.zarr"
            )
            csv_path = (
                data_root
                / "convolved_results"
                / "sector_0015_camera_1_ccd_4_removed_stars.csv"
            )
            convolved.mkdir(parents=True)
            (convolved / "cell_0_data").write_text("x", encoding="utf-8")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("source_id\n1\n", encoding="utf-8")

            cfg = RunnerConfig(data_root=str(data_root))
            resolved = resolve_config(target, cfg)
            removed = clear_ps1_process_artifacts(resolved)

            self.assertEqual(set(removed), {str(convolved), str(csv_path)})
            self.assertFalse(convolved.exists())
            self.assertFalse(csv_path.exists())
            self.assertEqual(clear_ps1_process_artifacts(resolved), [])


if __name__ == "__main__":
    unittest.main()
