"""Tests for template coverage and cropped template loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from syndiff_pipeline.common.template_coverage import (
    crop_bounds_subset_of_coverage,
    template_coverage_ffi_bounds,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    TemplateCoverageError,
    _load_ffi_cropped,
    _load_template_cropped,
)


class TestTemplateCoverage(unittest.TestCase):
    def _write_template(
        self, path: Path, shape: tuple[int, int], *, xmin=0, ymin=0
    ) -> None:
        ny, nx = shape
        data = np.ones(shape, dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data)
        hdu.header["XMIN"] = xmin
        hdu.header["YMIN"] = ymin
        hdu.header["XMAX"] = xmin + nx
        hdu.header["YMAX"] = ymin + ny
        hdu.writeto(path, overwrite=True)

    def test_full_chip_template_smaller_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl.fits"
            self._write_template(p, (2048, 2048))
            cov = template_coverage_ffi_bounds(str(p))
            crop = {"x_min": 512, "x_max": 1536, "y_min": 512, "y_max": 1536, "shape": (1024, 1024)}
            self.assertTrue(crop_bounds_subset_of_coverage(crop, cov))
            arr = _load_template_cropped(str(p), crop)
            self.assertEqual(arr.shape, (1024, 1024))

    def test_ffi_cropped_slice_matches_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ffi.fits"
            ny, nx = 512, 512
            sci = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
            err = np.ones((ny, nx), dtype=np.float32)
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(sci, name="SCI"),
                    fits.ImageHDU(err, name="ERR"),
                ]
            ).writeto(p, overwrite=True)
            bounds = {
                "x_min": 100,
                "x_max": 200,
                "y_min": 50,
                "y_max": 150,
                "shape": (100, 100),
            }
            sci_crop, err_crop = _load_ffi_cropped(str(p), bounds)
            self.assertEqual(sci_crop.shape, (100, 100))
            self.assertEqual(err_crop.shape, (100, 100))
            np.testing.assert_array_equal(sci_crop, sci[50:150, 100:200].astype(np.float64))

    def test_roi_template_crop_outside_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl.fits"
            self._write_template(p, (512, 512), xmin=100, ymin=100)
            crop = {"x_min": 0, "x_max": 512, "y_min": 0, "y_max": 512, "shape": (512, 512)}
            with self.assertRaises(TemplateCoverageError):
                _load_template_cropped(str(p), crop)


if __name__ == "__main__":
    unittest.main()
