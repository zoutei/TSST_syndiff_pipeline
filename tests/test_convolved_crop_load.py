"""Tests for loading convolved-template FITS written with FFI crop headers."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from astropy.io import fits

from syndiff_pipeline.common.fits_io import image_hdu_data, open_fits, write_image_fits
from syndiff_pipeline.difference_imaging.stages.kernel_subtract import _load_convolved_crop


def test_write_image_fits_strips_extname_for_primary():
    hdr = fits.Header()
    hdr["EXTNAME"] = "CAMERA.CCD 3.3 cal"
    hdr["NAXIS1"] = 4
    hdr["NAXIS2"] = 4
    arr = np.ones((4, 4), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_image_fits(f"{tmp}/conv.fits.fz", arr, header=hdr)
        with open_fits(path) as hdul:
            data = image_hdu_data(hdul)
            assert data.shape == (4, 4)


def test_load_convolved_crop_reads_extension_image():
    crop_bounds = {"shape": [4, 4], "x_min": 0, "x_max": 4, "y_min": 0, "y_max": 4}
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    hdr = fits.Header()
    hdr["EXTNAME"] = "CAMERA.CCD 3.3 cal"
    with tempfile.TemporaryDirectory() as tmp:
        plain = f"{tmp}/legacy.fits"
        fits.writeto(plain, arr, header=hdr, overwrite=True)
        loaded = _load_convolved_crop(plain, crop_bounds)
        np.testing.assert_allclose(loaded, arr.astype(np.float64))


def test_image_hdu_data_requires_2d():
    with pytest.raises(ValueError, match="No 2D image"):
        image_hdu_data(fits.HDUList([fits.PrimaryHDU()]))
