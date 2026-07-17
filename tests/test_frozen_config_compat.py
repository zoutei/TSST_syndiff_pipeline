"""Frozen run configs from newer feature branches must load on main."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import condor
from syndiff_pipeline.common.orchestration.state import PipelineState, STATUS_RUNNING
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.run_report import format_progress_lines
from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config
from syndiff_pipeline.template_creation.orchestration.stage_params import parse_stage_params


def _write_distortion_era_frozen_config(
    path: Path,
    *,
    workspace_root: str,
    data_root: str,
    runs_root: str,
    state_db_path: str,
) -> None:
    """Mirror keys from sn_multi_hp_epsf_20260712 that main does not model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"data_root: {data_root}",
                f"workspace_root: {workspace_root}",
                f"runs_root: {runs_root}",
                f"state_db_path: {state_db_path}",
                "stages:",
                "  wcs_grouping:",
                "    offset_threshold: 0.01",
                "    geometry_mode: linear",
                "    grouping_quantum_ps1_px: 1.0",
                "    drift_field:",
                "      grid_nx: 5",
                "      grid_ny: 5",
                "      include_corners: true",
                "      include_target: true",
                "      savgol_window: 11",
                "  mapping:",
                "    executor: condor",
                "  downsample:",
                "    materialize_templates: false",
                "  diff:",
                "    executor: condor",
                "  skycell_remap:",
                "    cache_quantum_ps1_px: 0.25",
                "    executor: condor",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class TestFrozenConfigCompat(unittest.TestCase):
    def test_strict_parse_accepts_geometry_mode(self):
        stages = parse_stage_params(
            {
                "wcs_grouping": {
                    "geometry_mode": "field",
                    "grouping_quantum_ps1_px": 1.0,
                },
                "downsample": {
                    "geometry_mode": "field",
                    "materialize_fits": False,
                    "hybrid_R": 1,
                    "apply_hybrid_exact": True,
                    "include_abutting_border_exact": True,
                    "rebuild_field_store": False,
                },
            }
        )
        self.assertEqual(stages.wcs_grouping.geometry_mode, "field")
        self.assertEqual(stages.downsample.geometry_mode, "field")
        self.assertFalse(stages.downsample.materialize_fits)
        self.assertTrue(stages.downsample.apply_hybrid_exact)
        self.assertTrue(stages.downsample.include_abutting_border_exact)
        self.assertFalse(stages.downsample.rebuild_field_store)

    def test_strict_parse_rejects_unknown_drift_field_block(self):
        with self.assertRaises(ValueError) as ctx:
            parse_stage_params(
                {
                    "wcs_grouping": {
                        "geometry_mode": "linear",
                        "drift_field": {"grid_nx": 5},
                        "grouping_quantum_ps1_px": 1.0,
                    }
                }
            )
        self.assertIn("drift_field", str(ctx.exception))

    def test_nonstrict_parse_drops_unknown_keys(self):
        stages = parse_stage_params(
            {
                "wcs_grouping": {
                    "offset_threshold": 0.02,
                    "geometry_mode": "field",
                    "drift_field": {"grid_nx": 5},
                    "grouping_quantum_ps1_px": 1.0,
                },
                "downsample": {"materialize_fits": False, "n_jobs": 8},
                "skycell_remap": {"executor": "condor"},
                "diff": {"executor": "condor"},
            },
            strict=False,
        )
        self.assertEqual(stages.wcs_grouping.offset_threshold, 0.02)
        self.assertEqual(stages.wcs_grouping.geometry_mode, "field")
        self.assertEqual(stages.downsample.n_jobs, 8)
        self.assertFalse(stages.downsample.materialize_fits)
        self.assertEqual(stages.diff.executor, "condor")

    def test_load_runner_config_accepts_distortion_era_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff = tmp_path / "handoff"
            runs_root = handoff / "runs"
            cfg_path = runs_root / "run_a" / "config.yaml"
            _write_distortion_era_frozen_config(
                cfg_path,
                workspace_root=str(handoff),
                data_root=str(tmp_path / "data"),
                runs_root=str(runs_root),
                state_db_path=str(handoff / "control" / "pipeline_state.sqlite"),
            )
            cfg = load_runner_config(cfg_path)
            self.assertEqual(cfg.stages.diff.executor, "condor")
            self.assertEqual(cfg.stages.wcs_grouping.offset_threshold, 0.01)

    def test_format_progress_lines_with_distortion_era_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff = tmp_path / "handoff"
            runs_root = handoff / "runs"
            run_dir = runs_root / "run_a"
            _write_distortion_era_frozen_config(
                run_dir / "config.yaml",
                workspace_root=str(handoff),
                data_root=str(tmp_path / "data"),
                runs_root=str(runs_root),
                state_db_path=str(handoff / "control" / "pipeline_state.sqlite"),
            )

            state = PipelineState(str(tmp_path / "state.sqlite"))
            target = Target(20, 3, 3, 221.0, 38.0, "2020ut")
            state.create_run(
                "run_a",
                str(run_dir / "config.yaml"),
                "/targets.csv",
                str(runs_root),
                [target],
                ["diff"],
            )
            label = target.label()
            state.update_stage_status("run_a", label, "diff", STATUS_RUNNING, started_at="t")
            # executor=None forces progress to reload frozen config for resolution
            state.set_launch_descriptor(
                "run_a",
                label,
                "diff",
                executor=None,
                native_id=42,
                submit_epoch=0.0,
                log_path=None,
            )
            with mock.patch.object(
                condor, "query_clusters_display", return_value={42: (condor._JOB_IDLE, None)}
            ):
                lines = format_progress_lines(state, "run_a", str(runs_root))
            self.assertTrue(any("status = " in line or "running=" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
