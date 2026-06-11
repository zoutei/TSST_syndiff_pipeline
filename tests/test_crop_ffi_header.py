"""Tests for SIP-safe FFI header cropping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from syndiff_pipeline.common import wcs_grouping


def _sip_ffi_header(nx: int = 200, ny: int = 200) -> fits.Header:
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["CTYPE1"] = "RA---TAN-SIP"
    hdr["CTYPE2"] = "DEC--TAN-SIP"
    hdr["CRVAL1"] = 100.0
    hdr["CRVAL2"] = 20.0
    hdr["CRPIX1"] = 100.0
    hdr["CRPIX2"] = 100.0
    hdr["CD1_1"] = -0.0001
    hdr["CD2_2"] = 0.0001
    hdr["A_ORDER"] = 2
    hdr["B_ORDER"] = 2
    hdr["A_1_0"] = 1e-7
    hdr["A_0_1"] = 1e-7
    hdr["B_1_0"] = 1e-7
    hdr["B_0_1"] = 1e-7
    hdr["DATE-OBS"] = "2020-01-01T00:00:00"
    hdr["CAMERA"] = 3
    hdr["CCD"] = 3
    return hdr


def _write_test_ffi(path: Path, hdr: fits.Header) -> None:
    data = np.zeros((int(hdr["NAXIS2"]), int(hdr["NAXIS1"])), dtype=np.float32)
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(data=data, header=hdr)
    fits.HDUList([primary, image]).writeto(path, overwrite=True)


class TestCropFfiHeader(unittest.TestCase):
    def test_crpix_naxis_and_sip_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffi_path = Path(tmp) / "test_ffi.fits"
            full_hdr = _sip_ffi_header()
            _write_test_ffi(ffi_path, full_hdr)

            crop_bounds = {
                "x_min": 20,
                "x_max": 120,
                "y_min": 30,
                "y_max": 130,
                "shape": (100, 100),
            }
            cropped = wcs_grouping.crop_ffi_header(str(ffi_path), crop_bounds)

            self.assertEqual(int(cropped["NAXIS1"]), 100)
            self.assertEqual(int(cropped["NAXIS2"]), 100)
            self.assertAlmostEqual(float(cropped["CRPIX1"]), 80.0)
            self.assertAlmostEqual(float(cropped["CRPIX2"]), 70.0)
            self.assertEqual(int(cropped["A_ORDER"]), 2)
            self.assertEqual(int(cropped["B_ORDER"]), 2)
            self.assertAlmostEqual(float(cropped["A_1_0"]), 1e-7)
            self.assertAlmostEqual(float(cropped["B_0_1"]), 1e-7)
            self.assertTrue(str(cropped["CTYPE1"]).endswith("-SIP"))
            self.assertEqual(cropped["DATE-OBS"], "2020-01-01T00:00:00")
            self.assertEqual(int(cropped["CAMERA"]), 3)
            self.assertEqual(int(cropped["XMIN"]), 20)
            self.assertEqual(int(cropped["ROIW"]), 100)

    def test_world_coords_match_full_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffi_path = Path(tmp) / "test_ffi.fits"
            full_hdr = _sip_ffi_header()
            _write_test_ffi(ffi_path, full_hdr)

            crop_bounds = {
                "x_min": 20,
                "x_max": 120,
                "y_min": 30,
                "y_max": 130,
                "shape": (100, 100),
            }
            cropped_hdr = wcs_grouping.crop_ffi_header(str(ffi_path), crop_bounds)
            w_full = WCS(full_hdr)
            w_crop = WCS(cropped_hdr)

            for cx, cy in [(1, 1), (50, 50), (100, 100)]:
                fx = cx + crop_bounds["x_min"]
                fy = cy + crop_bounds["y_min"]
                ra1, dec1 = w_full.all_pix2world(fx, fy, 0)
                ra2, dec2 = w_crop.all_pix2world(cx, cy, 0)
                self.assertAlmostEqual(ra1, ra2, places=9)
                self.assertAlmostEqual(dec1, dec2, places=9)


class TestWriteDiffNoiseMaskFitsHeader(unittest.TestCase):
    def test_primary_header_includes_sip(self):
        from syndiff_pipeline.difference_imaging.stages.hotpants import (
            write_diff_noise_mask_fits,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "diff.fits"
            hdr = _sip_ffi_header(nx=10, ny=10)
            hdr["CRPIX1"] = 5.0
            hdr["CRPIX2"] = 5.0
            hdr["NAXIS1"] = 10
            hdr["NAXIS2"] = 10
            diff = np.zeros((10, 10), dtype=np.float32)
            noise = np.ones((10, 10), dtype=np.float32)

            write_diff_noise_mask_fits(str(out_path), diff, noise, None, header=hdr)

            with fits.open(out_path) as hdul:
                self.assertEqual(int(hdul[0].header["A_ORDER"]), 2)
                self.assertTrue(str(hdul[0].header["CTYPE1"]).endswith("-SIP"))
                self.assertEqual(hdul[1].header["EXTNAME"], "NOISE")


if __name__ == "__main__":
    unittest.main()
