"""Tests for downsample input validation guards."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import zarr
from astropy.io import fits

from syndiff_pipeline.template_creation.processing.downsample import (
    PS1_REMOVED_STARS_CSV_FILENAME,
    create_syndiff_header,
    require_convolved_zarr_data,
    write_ps1_removed_star_gaia_csv,
)


class TestDownsampleValidation(unittest.TestCase):
    def test_require_convolved_zarr_data_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "sector_0040_camera_1_ccd_1.zarr"
            zarr.open(str(zarr_path), mode="w")
            with self.assertRaises(RuntimeError) as ctx:
                require_convolved_zarr_data(zarr_path)
            self.assertIn("empty", str(ctx.exception).lower())

    def test_require_convolved_zarr_data_with_data_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "sector_0040_camera_1_ccd_1.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            root["skycell.1234.567_data"] = np.ones((4, 4), dtype=np.float32)
            root["skycell.1234.567_mask"] = np.zeros((4, 4), dtype=np.uint32)
            require_convolved_zarr_data(zarr_path)

    def test_create_syndiff_header_copies_tess_reference_ffi_and_sector(self):
        tess_header = fits.Header()
        tess_header["TELESCOP"] = "TESS"
        tess_header["CAMERA"] = 3
        tess_header["CCD"] = 3
        tess_header["TESS_FFI"] = (
            "tess1234567890-s0020-cam3-ccd3-ff1-cad1-s0001.fits",
            "Reference FFI filename",
        )
        syndiff_header = create_syndiff_header(tess_header, sector=20)
        self.assertEqual(syndiff_header["SECTOR"], 20)
        self.assertEqual(syndiff_header["CAMERA"], 3)
        self.assertEqual(syndiff_header["CCD"], 3)
        self.assertEqual(
            syndiff_header["TESS_REFERENCE_FFI"],
            "tess1234567890-s0020-cam3-ccd3-ff1-cad1-s0001.fits",
        )
        self.assertNotIn("TESS_FFI", syndiff_header)
        self.assertLess(
            list(syndiff_header.keys()).index("SECTOR"),
            list(syndiff_header.keys()).index("CAMERA"),
        )
        self.assertLess(
            list(syndiff_header.keys()).index("CAMERA"),
            list(syndiff_header.keys()).index("CCD"),
        )

    def test_write_ps1_removed_star_gaia_csv_writes_event_dir_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_dir = tmp / "events" / "s0022_c3_c3_2020dgc"
            event_dir.mkdir(parents=True)

            ref_fits = tmp / "ref_ffi.fits"
            data = np.zeros((64, 64), dtype=np.float32)
            hdu0 = fits.PrimaryHDU()
            hdu1 = fits.ImageHDU(data=data)
            hdu1.header["NAXIS1"] = 64
            hdu1.header["NAXIS2"] = 64
            hdu1.header["CRPIX1"] = 32.0
            hdu1.header["CRPIX2"] = 32.0
            hdu1.header["CRVAL1"] = 228.0
            hdu1.header["CRVAL2"] = 52.0
            hdu1.header["CDELT1"] = -0.01
            hdu1.header["CDELT2"] = 0.01
            hdu1.header["CTYPE1"] = "RA---TAN"
            hdu1.header["CTYPE2"] = "DEC--TAN"
            fits.HDUList([hdu0, hdu1]).writeto(ref_fits, overwrite=True)

            job_path = event_dir / "cluster_template_job.json"
            job_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sector": 22,
                        "camera": 3,
                        "ccd": 3,
                        "x_min": 0,
                        "y_min": 0,
                        "x_max": 64,
                        "y_max": 64,
                        "reference_ffi_path": str(ref_fits),
                        "groups": [{"group_dx": 0.0, "group_dy": 0.0}],
                    }
                ),
                encoding="utf-8",
            )

            removed_csv = tmp / "removed_stars.csv"
            pd.DataFrame(
                [
                    {
                        "source_id": 1234567890123456789,
                        "ra": 228.0,
                        "dec": 52.0,
                        "tess_mag": 12.5,
                    }
                ]
            ).to_csv(removed_csv, index=False)

            out_path = write_ps1_removed_star_gaia_csv(
                job_json_path=job_path,
                removed_stars_csv=removed_csv,
                event_dir=event_dir,
                sector=22,
                camera=3,
                ccd=3,
                roi_bounds=(0, 0, 64, 64),
            )

            self.assertEqual(out_path, event_dir / PS1_REMOVED_STARS_CSV_FILENAME)
            self.assertTrue(out_path.is_file())
            out_df = pd.read_csv(out_path)
            self.assertIn("x", out_df.columns)
            self.assertIn("y", out_df.columns)
            self.assertEqual(len(out_df), 1)


if __name__ == "__main__":
    unittest.main()
