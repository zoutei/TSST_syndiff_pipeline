"""Tests for aperture forced photometry on difference-image cutouts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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


if __name__ == "__main__":
    unittest.main()
