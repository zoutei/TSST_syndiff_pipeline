"""Tests for downsample input validation guards."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import zarr
from astropy.io import fits

from syndiff_pipeline.template.downsample import create_syndiff_header, require_convolved_zarr_data


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


if __name__ == "__main__":
    unittest.main()
