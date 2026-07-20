"""Named remap/templates store_name resolution and debug PNG basenames."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    RunnerConfig,
    resolve_config,
)
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    RemapStageParams,
    TemplateStageParams,
    parse_stage_params,
)
from syndiff_pipeline.star.site_config import StarRunConfig
from syndiff_pipeline.template_creation.processing.shift_schedule_plots import (
    skycell_shift_debug_filenames,
)


def _minimal_stages(**kwargs) -> TemplateStageParams:
    stages = {
        "wcs_grouping": {},
        "mapping": {"oversampling_factor": 1},
        "ps1_download": {},
        "ps1_process": {},
        "remap": {},
        "downsample": {"geometry_mode": "field"},
    }
    for key, val in kwargs.items():
        stages[key] = {**stages.get(key, {}), **val}
    return parse_stage_params(stages)


class TestNamedStoreParams(unittest.TestCase):
    def test_downsample_rejects_both_apply_false(self):
        with self.assertRaises(ValueError):
            DownsampleStageParams(
                apply_intra_skycell=False, apply_inter_skycell=False
            )

    def test_invalid_store_name_raises(self):
        with self.assertRaises(ValueError):
            RemapStageParams(store_name="bad/name")
        with self.assertRaises(ValueError):
            DownsampleStageParams(output_store_name="..")

    def test_parse_named_lanes(self):
        stages = _minimal_stages(
            remap={"store_name": "hybrid_r2"},
            downsample={
                "output_store_name": "no_l4b",
                "remap_store_name": "hybrid_r2",
                "apply_inter_skycell": False,
            },
        )
        self.assertEqual(stages.remap.store_name, "hybrid_r2")
        self.assertEqual(stages.downsample.output_store_name, "no_l4b")
        self.assertEqual(stages.downsample.remap_store_name, "hybrid_r2")


class TestResolveNamedStores(unittest.TestCase):
    def _runner(self, tmp: Path, stages: TemplateStageParams) -> RunnerConfig:
        return RunnerConfig(
            data_root=str(tmp / "data"),
            workspace_root=str(tmp / "ws"),
            skycell_wcs_csv=str(tmp / "wcs.csv"),
            stages=stages,
        )

    def test_output_store_name_sets_template_output_base(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stages = _minimal_stages(
                downsample={"output_store_name": "no_l4b"},
            )
            target = Target(20, 1, 1, 0.0, 0.0, "x")
            resolved = resolve_config(target, self._runner(tmp, stages))
            self.assertTrue(
                resolved.template_output_base.endswith(
                    "templates_no_l4b/oversampling_1"
                )
            )
            self.assertTrue(resolved.remap_output_base.endswith("remap/oversampling_1"))
            self.assertIsNone(resolved.downsample_remap_store_name)

    def test_remap_store_name_inherits_remap_store_name(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stages = _minimal_stages(
                remap={"store_name": "hybrid_r2"},
                downsample={"output_store_name": "hybrid_r2"},
            )
            target = Target(20, 1, 1, 0.0, 0.0, "x")
            resolved = resolve_config(target, self._runner(tmp, stages))
            self.assertEqual(resolved.downsample_remap_store_name, "hybrid_r2")
            self.assertTrue(
                resolved.remap_output_base.endswith("remap_hybrid_r2/oversampling_1")
            )
            self.assertTrue(
                resolved.template_output_base.endswith(
                    "templates_hybrid_r2/oversampling_1"
                )
            )

    def test_explicit_remap_store_name_overrides_inherit(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stages = _minimal_stages(
                remap={"store_name": "hybrid_r2"},
                downsample={
                    "output_store_name": "no_l4b",
                    "remap_store_name": "other",
                },
            )
            target = Target(20, 1, 1, 0.0, 0.0, "x")
            resolved = resolve_config(target, self._runner(tmp, stages))
            self.assertEqual(resolved.downsample_remap_store_name, "other")


class TestDebugFilenames(unittest.TestCase):
    def test_default_and_named(self):
        grid, rel = skycell_shift_debug_filenames(None)
        self.assertEqual(grid, "skycell_shift_grid_debug.png")
        self.assertEqual(rel, "skycell_shift_relative_center_debug.png")
        grid2, rel2 = skycell_shift_debug_filenames("hybrid_r2")
        self.assertEqual(grid2, "skycell_shift_grid_debug_hybrid_r2.png")
        self.assertEqual(rel2, "skycell_shift_relative_center_debug_hybrid_r2.png")


class TestStarTemplateStoreName(unittest.TestCase):
    def test_star_run_config_carries_template_store_name(self):
        cfg = StarRunConfig(template_store_name="no_l4b", oversampling_factor=2)
        self.assertEqual(cfg.template_store_name, "no_l4b")
        self.assertEqual(cfg.oversampling_factor, 2)


if __name__ == "__main__":
    unittest.main()
