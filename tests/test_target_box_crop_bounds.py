"""Tests for target_box crop mode and shared crop resolvers."""

from __future__ import annotations

import unittest

from astropy.io import fits
import numpy as np

from syndiff_pipeline.common import wcs_grouping


def _ffi_header(nx: int = 2048, ny: int = 2048) -> fits.Header:
    data = np.zeros((ny, nx), dtype=np.float32)
    hdu1 = fits.ImageHDU(data=data)
    hdu1.header["NAXIS1"] = nx
    hdu1.header["NAXIS2"] = ny
    hdu1.header["CRPIX1"] = nx / 2.0
    hdu1.header["CRPIX2"] = ny / 2.0
    hdu1.header["CRVAL1"] = 100.0
    hdu1.header["CRVAL2"] = 10.0
    hdu1.header["CDELT1"] = -0.001
    hdu1.header["CDELT2"] = 0.001
    hdu1.header["CTYPE1"] = "RA---TAN"
    hdu1.header["CTYPE2"] = "DEC--TAN"
    return hdu1.header


class TestTargetBoxCropBounds(unittest.TestCase):
    def test_centered_1024_box(self):
        hdr = _ffi_header()
        bounds = wcs_grouping.get_target_box_crop_bounds(
            hdr, 100.0, 10.0, box_size=1024
        )
        self.assertEqual(bounds["shape"], (1024, 1024))
        self.assertAlmostEqual(
            (bounds["x_min"] + bounds["x_max"]) / 2, 1024, delta=2
        )
        self.assertAlmostEqual(
            (bounds["y_min"] + bounds["y_max"]) / 2, 1024, delta=2
        )

    def test_box_stays_inside_chip(self):
        hdr = _ffi_header()
        bounds = wcs_grouping.get_target_box_crop_bounds(
            hdr, 100.0, 10.0, box_size=1024
        )
        nx = int(hdr["NAXIS1"])
        ny = int(hdr["NAXIS2"])
        self.assertGreaterEqual(bounds["x_min"], 0)
        self.assertGreaterEqual(bounds["y_min"], 0)
        self.assertLessEqual(bounds["x_max"], nx)
        self.assertLessEqual(bounds["y_max"], ny)

    def test_manual_box_wins_over_target_box(self):
        hdr = _ffi_header()
        bounds = wcs_grouping.resolve_crop_bounds_from_params(
            hdr,
            x_min=10,
            x_max=110,
            y_min=20,
            y_max=120,
            crop_mode="target_box",
            target_ra=100.0,
            target_dec=10.0,
        )
        self.assertEqual(bounds["x_min"], 10)
        self.assertEqual(bounds["x_max"], 110)

    def test_target_off_chip_raises(self):
        hdr = _ffi_header()
        with self.assertRaises(ValueError):
            wcs_grouping.get_target_box_crop_bounds(
                hdr, 50.0, 50.0, box_size=1024
            )

    def test_diff_crop_explicit_default_full_not_override(self):
        class Cfg:
            x_min = x_max = y_min = y_max = None
            crop_mode = None

        self.assertFalse(wcs_grouping.diff_crop_explicitly_configured(Cfg()))

    def test_diff_crop_explicit_target_box(self):
        class Cfg:
            x_min = x_max = y_min = y_max = None
            crop_mode = "target_box"

        self.assertTrue(wcs_grouping.diff_crop_explicitly_configured(Cfg()))

    def test_diff_crop_explicit_quadrant_override(self):
        class Cfg:
            x_min = x_max = y_min = y_max = None
            crop_mode = "tl"

        self.assertTrue(wcs_grouping.diff_crop_explicitly_configured(Cfg()))

    def test_resolve_crop_mode_quadrant(self):
        hdr = _ffi_header()
        bounds = wcs_grouping.resolve_crop_bounds_from_params(hdr, crop_mode="tl")
        self.assertLess(bounds["x_max"], int(hdr["NAXIS1"]) // 2 + 50)


if __name__ == "__main__":
    unittest.main()
