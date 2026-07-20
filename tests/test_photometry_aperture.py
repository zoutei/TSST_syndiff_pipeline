"""Tests for aperture forced photometry on difference-image cutouts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    AperturePhotometryMethodParams,
)
from syndiff_pipeline.difference_imaging.stages import photometry as ph


class TestApertureFluxOnCutout(unittest.TestCase):
    def test_flux_wo_sky_equals_flux_minus_sky(self):
        size = 11
        tar_ap = 3
        sky_in = 5
        sky_out = 9
        half = size // 2
        ap_tar, ap_sky, n_tar = ph._build_aperture_masks(
            (size, size), half, half, tar_ap, sky_in, sky_out
        )
        data = np.full((size, size), 2.0, dtype=np.float64)
        data[half, half] = 20.0
        flux, sky, flux_wo_sky, eflux = ph.aperture_flux_on_cutout(
            data, ap_tar, ap_sky, n_tar, sigma=None, sky_mask=None
        )
        self.assertAlmostEqual(flux_wo_sky, flux - sky)
        self.assertTrue(np.isfinite(eflux))
        self.assertGreater(flux, flux_wo_sky)


class TestApertureSharedMaskAndSubtractSky(unittest.TestCase):
    def test_combine_sky_mask_with_shared_bits(self):
        sky = np.zeros((5, 5), dtype=bool)
        sky[2, 2] = True
        shared = np.zeros((5, 5), dtype=np.int16)
        shared[1, 1] = 1  # catalog bit
        shared[3, 3] = 2  # bright-cross bit
        shared[4, 4] = 4  # strap — ignored by default
        out = ph._combine_sky_mask_with_shared(sky, shared)
        self.assertTrue(out[2, 2])
        self.assertTrue(out[1, 1])
        self.assertTrue(out[3, 3])
        self.assertFalse(out[4, 4])

    def test_subtract_sky_false_uses_raw_flux_for_zp(self):
        n_ep = 2
        paths = [f"/fake/diff_{i}.fits" for i in range(n_ep)]
        wcs = pd.DataFrame({"btjd": [100.0, 101.0], "group_id": [0, 0]})
        xy = np.full((n_ep, 2), 16.0, dtype=np.float64)
        method = AperturePhotometryMethodParams(
            name="ap3",
            tar_ap=3,
            sky_in=5,
            sky_out=9,
            subtract_sky=False,
        )
        targets = [
            ph.ForcedPhotTargetSpec(
                target_xy=xy,
                csv_basename="lightcurve_ap3.csv",
                plot_source_label="primary",
                tag="primary",
            )
        ]
        cfg = SimpleNamespace(
            sector=20, camera=3, ccd=3, n_jobs=1, pipeline_plots=False, pipeline_plot_dpi=150
        )
        data = np.full((32, 32), 1.0, dtype=np.float64)
        data[16, 16] = 50.0
        captured = {}

        def _fake_zp(lc_df, diffs_dir, flux_col="flux", eflux_col="eflux"):
            captured["flux_col"] = flux_col
            return lc_df

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                ph, "read_diff_primary_and_noise_sigma", return_value=(data, None)
            ):
                with patch.object(ph.os.path, "exists", return_value=True):
                    with patch.object(ph, "apply_zp_calibration_if_available", side_effect=_fake_zp):
                        ph._run_aperture_photometry_multi(
                            diff_paths=paths,
                            targets=targets,
                            method=method,
                            wcs_table=wcs,
                            cfg=cfg,
                            output_dir=tmp,
                            diffs_dir=tmp,
                        )
        self.assertEqual(captured["flux_col"], "flux")

    def test_shared_mask_excludes_sky_annulus_pixels(self):
        size = 11
        half = size // 2
        ap_tar, ap_sky, n_tar = ph._build_aperture_masks(
            (size, size), half, half, 3, 5, 9
        )
        data = np.full((size, size), 2.0, dtype=np.float64)
        data[half, half] = 20.0
        # Mild bias across the whole annulus (not a single outlier that σ-clip removes).
        annulus = np.isfinite(ap_sky) & (ap_sky != 0)
        data[annulus] = 10.0

        shared = np.zeros((size, size), dtype=np.int16)
        shared[annulus] = 1

        _, sky0, _, _ = ph.aperture_flux_on_cutout(
            data, ap_tar, ap_sky, n_tar, sky_mask=None
        )
        sky_mask = ph._combine_sky_mask_with_shared(None, shared)
        _, sky1, _, _ = ph.aperture_flux_on_cutout(
            data, ap_tar, ap_sky, n_tar, sky_mask=sky_mask
        )
        self.assertAlmostEqual(sky0, 10.0 * n_tar)
        # Masked annulus → no finite sky pixels → NaN sky contribution.
        self.assertTrue(np.isnan(sky1) or sky1 != sky0)


if __name__ == "__main__":
    unittest.main()
