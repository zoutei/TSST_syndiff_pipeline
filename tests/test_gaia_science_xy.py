"""Tests for per-frame Gaia → science-array coordinate projection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.wcs_header_cache import extract_ffi_header_record


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
    return hdr


def _write_test_ffi(path: Path, hdr: fits.Header) -> None:
    data = np.zeros((int(hdr["NAXIS2"]), int(hdr["NAXIS1"])), dtype=np.float32)
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(data=data, header=hdr)
    fits.HDUList([primary, image]).writeto(path, overwrite=True)


class TestGaiaScienceXy(unittest.TestCase):
    def test_matches_crop_ffi_header_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ffi_path = Path(tmp) / "tess_s0020_3_3_ffic.fits"
            full_hdr = _sip_ffi_header()
            _write_test_ffi(ffi_path, full_hdr)

            science_bounds = {
                "x_min": 20,
                "x_max": 120,
                "y_min": 0,
                "y_max": 130,
                "shape": (130, 100),
            }
            w_full = WCS(full_hdr)
            w_crop = WCS(wcs_grouping.crop_ffi_header(str(ffi_path), science_bounds))

            fx, fy = 100.0, 80.0
            ra0, dec0 = w_full.all_pix2world(fx, fy, 0)
            gaia = pd.DataFrame(
                {
                    "source_id": [1, 2],
                    "ra": [ra0, ra0 + 0.001],
                    "dec": [dec0, dec0 + 0.001],
                    "phot_rp_mean_mag": [10.0, 11.0],
                    "x": [999.0, 888.0],
                    "y": [666.0, 555.0],
                }
            )
            parent_x = gaia["x"].copy()

            row = extract_ffi_header_record(
                ffi_path, open_fits=wcs_grouping.open_fits_memmap
            )
            ffi_list_df = pd.DataFrame([row]).set_index("filename")

            out = wcs_grouping.gaia_science_xy_for_frame(
                gaia, str(ffi_path), ffi_list_df, science_bounds
            )
            self.assertFalse(out.empty)
            pd.testing.assert_series_equal(gaia["x"], parent_x)

            for _, star in out.iterrows():
                ra2, dec2 = w_crop.all_pix2world(star["x"], star["y"], 0)
                self.assertAlmostEqual(star["ra"], ra2, places=6)
                self.assertAlmostEqual(star["dec"], dec2, places=6)

    def test_requires_ra_dec(self):
        gaia = pd.DataFrame({"x": [1.0], "y": [2.0]})
        bounds = {"x_min": 0, "y_min": 0, "shape": (10, 10)}
        with self.assertRaises(ValueError):
            wcs_grouping.gaia_science_xy_for_frame(
                gaia, "/fake.fits", pd.DataFrame(), bounds
            )


if __name__ == "__main__":
    unittest.main()
