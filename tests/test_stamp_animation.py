"""Tests for per-epoch stamp GIF helpers (forced photometry debug plots)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.support import plot as plot_mod
from syndiff_pipeline.difference_imaging.stages import photometry as ph


class TestFixedScaleLimits(unittest.TestCase):
    def test_symmetric_scale(self):
        cube = np.array([[[-2.0, 0.0], [0.5, 1.0]]], dtype=float)
        vmin, vmax = plot_mod._fixed_scale_limits(cube, "symmetric")
        self.assertLess(vmin, 0.0)
        self.assertGreater(vmax, 0.0)
        self.assertAlmostEqual(vmin, -vmax)

    def test_percentile_scale(self):
        cube = np.arange(100, dtype=float).reshape(1, 10, 10)
        vmin, vmax = plot_mod._fixed_scale_limits(cube, "percentile")
        self.assertLess(vmin, vmax)
        self.assertGreaterEqual(vmin, 0.0)


class TestStampAnimations(unittest.TestCase):
    def test_write_stamp_and_dual_gifs(self):
        size = 15
        n = 4
        rng = np.random.default_rng(0)
        diff = [rng.normal(0, 1, (size, size)) for _ in range(n)]
        sci = [rng.uniform(100, 200, (size, size)) for _ in range(n)]
        btjd = np.array([2459000.1, 2459000.2, 2459000.3, 2459000.4])

        with tempfile.TemporaryDirectory() as tmp:
            diff_path = os.path.join(tmp, "cutout_diff.gif")
            sci_path = os.path.join(tmp, "cutout_science.gif")
            pair_path = os.path.join(tmp, "cutout_pair.gif")

            self.assertIsNotNone(
                plot_mod.write_stamp_animation(
                    diff,
                    diff_path,
                    btjd=btjd,
                    stamp_size=size,
                    scale_mode="symmetric",
                )
            )
            self.assertIsNotNone(
                plot_mod.write_stamp_animation(
                    sci,
                    sci_path,
                    btjd=btjd,
                    stamp_size=size,
                    cmap="viridis",
                    scale_mode="percentile",
                    cbar_label="Science stamp",
                )
            )
            self.assertIsNotNone(
                plot_mod.write_dual_stamp_animation(
                    diff, sci, pair_path, btjd=btjd, stamp_size=size
                )
            )
            for p in (diff_path, sci_path, pair_path):
                self.assertTrue(os.path.isfile(p))
                self.assertGreater(os.path.getsize(p), 100)


class TestScienceCutoutExtraction(unittest.TestCase):
    def test_extract_science_cutout_from_ffi(self):
        ny, nx = 64, 64
        data = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
        crop_bounds = {"x_min": 10, "x_max": 50, "y_min": 5, "y_max": 45}

        with tempfile.TemporaryDirectory() as tmp:
            ffi_path = os.path.join(tmp, "test_ffic.fits")
            hdu = fits.PrimaryHDU()
            hdu1 = fits.ImageHDU(data)
            hdu2 = fits.ImageHDU(np.ones_like(data))
            fits.HDUList([hdu, hdu1, hdu2]).writeto(ffi_path, overwrite=True)

            wcs = pd.DataFrame({"path": [ffi_path], "btjd": [100.0]})
            target_xy = np.array([[20.0, 20.0]], dtype=np.float64)
            cutouts = ph._extract_science_cutouts_for_epochs(
                wcs, target_xy, crop_bounds, phot_cutout_size=15, n_jobs=1
            )
            self.assertEqual(len(cutouts), 1)
            self.assertIsNotNone(cutouts[0])
            self.assertEqual(cutouts[0].shape, (15, 15))


if __name__ == "__main__":
    unittest.main()
