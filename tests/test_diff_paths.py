"""Tests for difference-imaging workspace path helpers."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_templates_symlink,
    event_templates_symlink_path,
)
from syndiff_pipeline.difference_imaging.orchestration.config import (
    SynDiffConfig,
    absolutize_config,
)
from syndiff_pipeline.difference_imaging.stages.photometry import (
    write_lightcurve_diagnostic_plot,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    KERNEL_RECONSTRUCTION_NPZ_BASENAME,
    clear_diff_workspace,
    meta_workspace_dir_from_diffs_dir,
    meta_workspace_label,
    normalize_photometry_run_id,
    photometry_root,
    photometry_tree_name,
    pipeline_plots_root,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    kernel_reconstruction_npz_path,
)


class TestPipelinePlotsRoot(unittest.TestCase):
    def test_default_subdir_under_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = os.path.join(tmp, "events", "s0020_c3_k3_2020ut")
            self.assertEqual(
                pipeline_plots_root(event),
                os.path.join(os.path.abspath(event), "ws", "debug_plots"),
            )

    def test_empty_subdir_returns_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = os.path.join(tmp, "event")
            self.assertEqual(
                pipeline_plots_root(event, ""),
                os.path.join(os.path.abspath(event), "ws"),
            )

    def test_absolutize_config_keeps_pipeline_plots_dir_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "config"
            site.mkdir()
            cfg = SynDiffConfig(
                output_dir=str(Path(tmp) / "events" / "test"),
                pipeline_plots_dir="debug_plots",
            )
            frozen = absolutize_config(cfg, site)
            self.assertEqual(frozen.pipeline_plots_dir, "debug_plots")
            self.assertEqual(
                pipeline_plots_root(frozen.output_dir, frozen.pipeline_plots_dir),
                str((Path(tmp) / "events" / "test" / "ws" / "debug_plots").resolve()),
            )


class TestLightcurveDiagnosticPlot(unittest.TestCase):
    def test_writes_empty_plot_when_all_flux_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                {
                    "btjd": [100.0, 101.0, 102.0],
                    "flux": [float("nan")] * 3,
                    "eflux": [float("nan")] * 3,
                }
            )
            out = os.path.join(tmp, "lc_empty.png")
            path = write_lightcurve_diagnostic_plot(df, tmp, png_path=out, title_line="x")
            self.assertEqual(path, out)
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 1000)


class TestPhotometryPaths(unittest.TestCase):
    def test_photometry_tree_name_default(self):
        self.assertEqual(photometry_tree_name(), "phot")
        self.assertEqual(photometry_tree_name(None), "phot")

    def test_photometry_tree_name_with_run_id(self):
        self.assertEqual(photometry_tree_name("debug1"), "phot_debug1")

    def test_photometry_root_under_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = os.path.join(tmp, "events", "s0020_c3_k3_2020ut")
            self.assertEqual(
                photometry_root(event),
                os.path.join(os.path.abspath(event), "phot"),
            )
            self.assertEqual(
                photometry_root(event, "smoke"),
                os.path.join(os.path.abspath(event), "phot_smoke"),
            )

    def test_normalize_photometry_run_id(self):
        self.assertIsNone(normalize_photometry_run_id(None))
        self.assertIsNone(normalize_photometry_run_id(""))
        self.assertEqual(normalize_photometry_run_id("run_a"), "run_a")


class TestClearDiffWorkspace(unittest.TestCase):
    def test_clear_diff_workspace_restores_templates_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            physical = Path(tmp) / "templates_physical"
            physical.mkdir()
            ensure_event_templates_symlink(out, physical)
            (out / "ws" / "hp_d").mkdir(parents=True)
            (out / "ws" / "hp_d" / "x.fits").write_bytes(b"x")

            clear_diff_workspace(out)
            link = event_templates_symlink_path(out)
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), physical.resolve())


class TestMetaWorkspaceLabel(unittest.TestCase):
    def test_diffs_to_meta(self):
        self.assertEqual(meta_workspace_label("hp_d"), "hp_m")
        self.assertEqual(meta_workspace_label("ks_d"), "ks_m")

    def test_non_d_suffix(self):
        self.assertEqual(meta_workspace_label("diff_r1"), "diff_r1_m")

    def test_kernel_reconstruction_under_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            diffs = os.path.join(tmp, "ws", "hp_d")
            os.makedirs(diffs)
            path = kernel_reconstruction_npz_path(diffs)
            self.assertEqual(
                path,
                os.path.join(tmp, "ws", "hp_m", KERNEL_RECONSTRUCTION_NPZ_BASENAME),
            )
            self.assertEqual(
                meta_workspace_dir_from_diffs_dir(diffs),
                os.path.join(tmp, "ws", "hp_m"),
            )


if __name__ == "__main__":
    unittest.main()
