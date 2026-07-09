"""Tests for star/plots.py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.star.identifiers import ResolvedHost
from syndiff_pipeline.star.plots import (
    write_lightcurve_debug_png,
    write_mini_template_downsample_png,
    write_ps1_segment_overlay_png,
)


def _host() -> ResolvedHost:
    return ResolvedHost(
        input_kind="gaia",
        input_value=1060421588522505216,
        tic_id=None,
        gaia_source_id=1060421588522505216,
        ra=120.0,
        dec=30.0,
        phot_g_mean_mag=12.0,
        phot_bp_mean_mag=12.2,
        phot_rp_mean_mag=11.8,
        resolution_method="test",
        label=None,
    )


class TestStarPlots(unittest.TestCase):
    def test_ps1_segment_overlay_png_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ps1_segment_skycell.1.2.png"
            bg = np.random.default_rng(0).normal(10.0, 1.0, (64, 64)).astype(np.float32)
            bg = np.clip(bg, 0.1, None)
            mask = np.zeros((64, 64), dtype=bool)
            mask[28:36, 28:36] = True

            overlay = np.zeros_like(bg)
            overlay[mask] = bg[mask]
            path = write_ps1_segment_overlay_png(
                out,
                original_image=bg,
                data_wo_bkg_sat=overlay,
                host_pixel_xy=(32.0, 32.0),
                skycell_name="skycell.1.2",
                host=_host(),
                blend_flag=False,
                cutout_bounds=(0, 0, 64, 64),
                target_seg_id=3,
            )

            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 1000)

    def test_ps1_segment_overlay_no_segment_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ps1_segment_none.png"
            bg = np.full((32, 32), 1.0, dtype=np.float32)

            path = write_ps1_segment_overlay_png(
                out,
                original_image=bg,
                data_wo_bkg_sat=np.zeros_like(bg),
                host_pixel_xy=(16.0, 16.0),
                skycell_name="skycell.1.2",
                host=_host(),
                blend_flag=False,
                cutout_bounds=(0, 0, 32, 32),
                target_seg_id=0,
            )

            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 500)

    def test_mini_template_downsample_png_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mini_template.png"
            flux = np.zeros((48, 48), dtype=np.float32)
            flux[22:26, 22:26] = 50.0
            production = np.zeros((64, 64), dtype=np.float32)
            production[30:34, 30:34] = 40.0

            path = write_mini_template_downsample_png(
                out,
                mini_flux_sum=flux,
                host_local_xy=(24.0, 24.0),
                dx=0.0,
                dy=0.0,
                host=_host(),
                production_template_slice=production,
                roi_bounds=(10, 10, 58, 58),
            )

            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 1000)

    def test_lightcurve_debug_png_single_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "lc.png"
            df = pd.DataFrame(
                {
                    "btjd": [1842.5, 1842.52, 1842.54],
                    "flux": [100.0, 102.0, 99.0],
                    "flux_wo_sky": [100.0, 102.0, 99.0],
                    "eflux": [1.0, 1.1, 0.9],
                }
            )
            path = write_lightcurve_debug_png(
                out, lightcurves={"ap3": df}, host=_host(),
            )
            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 1000)

    def test_lightcurve_debug_png_multi_method_no_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "lc_multi.png"
            ap = pd.DataFrame(
                {
                    "btjd": [np.nan, np.nan],
                    "flux": [10.0, 12.0],
                    "flux_wo_sky": [10.0, 12.0],
                    "eflux": [0.5, 0.6],
                }
            )
            prf = pd.DataFrame(
                {
                    "btjd": [np.nan, np.nan],
                    "flux": [-5.0, -6.0],
                    "eflux": [0.2, 0.3],
                }
            )
            path = write_lightcurve_debug_png(
                out, lightcurves={"ap3": ap, "prf": prf}, host=_host(),
            )
            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 1000)

    def test_lightcurve_debug_png_raises_on_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "lc_empty.png"
            with self.assertRaises(ValueError):
                write_lightcurve_debug_png(
                    out, lightcurves={"ap3": pd.DataFrame()}, host=_host(),
                )


if __name__ == "__main__":
    unittest.main()
