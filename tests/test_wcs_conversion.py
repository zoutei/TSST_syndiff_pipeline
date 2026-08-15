"""Tests for syndiff_pipeline.difference_imaging.wcs.wcs_conversion."""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.wcs.wcs_conversion import (
    cd_matrix_from_header,
    radec_to_uv,
)


def _tesslike_header() -> fits.Header:
    w = WCS(naxis=2)
    w.wcs.crpix = [512.0, 512.0]
    w.wcs.crval = [120.0, 45.0]
    w.wcs.cd = [[-0.00028, 0.0], [0.0, 0.00028]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = w.to_header()
    hdr["A_ORDER"] = 2
    hdr["B_ORDER"] = 2
    hdr["A_0_2"] = 1.0e-7
    hdr["B_2_0"] = -8.0e-8
    return hdr


def test_cd_matrix_from_pc_cdelt():
    hdr = fits.Header()
    hdr["PC1_1"] = 1.0
    hdr["PC1_2"] = 0.0
    hdr["PC2_1"] = 0.0
    hdr["PC2_2"] = 1.0
    hdr["CDELT1"] = -0.001
    hdr["CDELT2"] = 0.001
    cd = cd_matrix_from_header(hdr)
    assert np.allclose(cd, np.diag([-0.001, 0.001]))


def test_radec_to_uv_matches_astropy_world_to_pixel():
    hdr = _tesslike_header()
    wcs = WCS(hdr)
    ra = np.array([120.01, 119.99, 120.0])
    dec = np.array([45.01, 44.99, 45.0])
    u, v = radec_to_uv(ra, dec, hdr)
    x, y = wcs.world_to_pixel_values(ra, dec)
    crpix1 = float(hdr["CRPIX1"])
    crpix2 = float(hdr["CRPIX2"])
    expected_u = x - (crpix1 - 1.0)
    expected_v = y - (crpix2 - 1.0)
    assert np.allclose(u, expected_u, atol=1e-6)
    assert np.allclose(v, expected_v, atol=1e-6)
