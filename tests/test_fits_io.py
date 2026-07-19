"""Tests for fpack-based FITS writes and open_fits primary promotion."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.fits_io import open_fits, write_hdul_fits, write_image_fits


@unittest.skipUnless(shutil.which("fpack"), "fpack not on PATH")
class TestFitsIo(unittest.TestCase):
    def test_write_image_round_trip_and_no_plain_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
            out = write_image_fits(Path(tmp) / "img.fits.gz", data)
            self.assertTrue(out.endswith(".fits.fz"))
            self.assertTrue(os.path.isfile(out))
            self.assertFalse(os.path.isfile(Path(tmp) / "img.fits"))
            self.assertFalse(os.path.isfile(Path(tmp) / "img.fits.gz"))
            with open_fits(out) as hdul:
                self.assertIsNotNone(hdul[0].data)
                np.testing.assert_array_equal(hdul[0].data, data)

    def test_multi_hdu_preserves_named_exts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.ones((3, 3), dtype=np.float32)
            hdul = fits.HDUList(
                [
                    fits.PrimaryHDU(data),
                    fits.ImageHDU(data * 2, name="NOISE"),
                    fits.ImageHDU(data * 3, name="MASK"),
                ]
            )
            out = write_hdul_fits(Path(tmp) / "diff.fits", hdul)
            with open_fits(out) as h:
                np.testing.assert_array_equal(h[0].data, data)
                np.testing.assert_array_equal(h["NOISE"].data, data * 2)
                np.testing.assert_array_equal(h["MASK"].data, data * 3)

    def test_legacy_gz_still_opens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.fits.gz"
            data = np.eye(2, dtype=np.float32)
            fits.writeto(path, data, overwrite=True)
            with open_fits(Path(tmp) / "legacy.fits.fz") as hdul:
                np.testing.assert_array_equal(hdul[0].data, data)


if __name__ == "__main__":
    unittest.main()
