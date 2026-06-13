"""Tests for ensure_gaia_crop_xy WCS re-projection."""

from __future__ import annotations

import tempfile
import unittest

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd

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


def _write_ref_ffi(path: str, nx: int = 2048, ny: int = 2048) -> fits.Header:
    hdr = _ffi_header(nx=nx, ny=ny)
    data = np.zeros((ny, nx), dtype=np.float32)
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(data=data, header=hdr)
    fits.HDUList([primary, image]).writeto(path, overwrite=True)
    return hdr


class TestEnsureGaiaCropXy(unittest.TestCase):
    def test_stale_crop_xy_reprojected_from_ra_dec(self):
        target_ra, target_dec = 100.0, 10.0
        star_ra, star_dec = 100.05, 10.02

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = f"{tmp}/ref_ffi.fits"
            hdr = _write_ref_ffi(ref_path)
            crop_bounds = wcs_grouping.get_target_box_crop_bounds(
                hdr, target_ra, target_dec, box_size=512
            )

            gaia_df = pd.DataFrame(
                {
                    "source_id": [12345],
                    "ra": [star_ra],
                    "dec": [star_dec],
                    "x": [9999.0],
                    "y": [9999.0],
                }
            )

            out = wcs_grouping.ensure_gaia_crop_xy(
                gaia_df, ref_path, crop_bounds
            )

            wcs = WCS(hdr)
            x_ffi, y_ffi = wcs_grouping.world_ra_dec_to_pixel(wcs, star_ra, star_dec)
            expected_x = x_ffi - crop_bounds["x_min"]
            expected_y = y_ffi - crop_bounds["y_min"]

            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(float(out.iloc[0]["x"]), expected_x, places=3)
            self.assertAlmostEqual(float(out.iloc[0]["y"]), expected_y, places=3)
            self.assertGreaterEqual(float(out.iloc[0]["x"]), 0.0)
            self.assertGreaterEqual(float(out.iloc[0]["y"]), 0.0)
            ny, nx = crop_bounds["shape"]
            self.assertLess(float(out.iloc[0]["x"]), nx)
            self.assertLess(float(out.iloc[0]["y"]), ny)


if __name__ == "__main__":
    unittest.main()
