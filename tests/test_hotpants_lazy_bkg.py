"""Tests for lazy per-frame sci_bkg loading in hotpants."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages import hotpants


class TestLoadSciBkgCrop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bkg_ws = Path(self._tmp.name) / "ks_b"
        self.bkg_ws.mkdir()
        self.shape = (32, 32)
        self.product_id = "tess2026039233236"

    def tearDown(self):
        self._tmp.cleanup()

    def _write_bkg(self, data: np.ndarray) -> None:
        stem = hotpants.workspace_frame_stem(self.product_id, "ks_b")
        path = self.bkg_ws / f"{stem}.fits"
        fits.writeto(path, data.astype(np.float32), overwrite=True)

    def test_loads_existing_fits(self):
        data = np.ones(self.shape, dtype=np.float64) * 3.5
        self._write_bkg(data)
        out = hotpants._load_sci_bkg_crop(str(self.bkg_ws), self.product_id, self.shape)
        np.testing.assert_array_equal(out, data)

    def test_missing_file_returns_zeros(self):
        out = hotpants._load_sci_bkg_crop(str(self.bkg_ws), self.product_id, self.shape)
        self.assertEqual(out.shape, self.shape)
        self.assertTrue(np.all(out == 0))

    def test_shape_mismatch_returns_zeros(self):
        self._write_bkg(np.zeros((16, 16), dtype=np.float64))
        out = hotpants._load_sci_bkg_crop(str(self.bkg_ws), self.product_id, self.shape)
        self.assertEqual(out.shape, self.shape)
        self.assertTrue(np.all(out == 0))


if __name__ == "__main__":
    unittest.main()
