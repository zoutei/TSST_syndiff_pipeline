"""Tests for manifest fast path in per_frame_target_crop_xy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages import photometry


class TestPerFrameTargetCropXYManifest(unittest.TestCase):
    def test_uses_manifest_columns_for_science_target(self):
        wcs = pd.DataFrame(
            {
                "path": ["a.fits", "b.fits"],
                "x_pix": [300.0, 305.0],
                "y_pix": [400.0, 402.0],
                "wcs_ok": [True, True],
            }
        )
        crop = {"x_min": 267, "y_min": 0, "shape": (1024, 1024)}
        science = (169.9612188, -27.8636818)

        with mock.patch(
            "syndiff_pipeline.common.wcs_grouping.open_fits_memmap",
        ) as open_fits:
            out = photometry.per_frame_target_crop_xy(
                wcs,
                science[0],
                science[1],
                crop,
                manifest_science_ra_dec=science,
            )
            open_fits.assert_not_called()

        np.testing.assert_allclose(out, [[33.0, 400.0], [38.0, 402.0]])

    def test_reprojects_other_sky_positions_via_fits(self):
        wcs = pd.DataFrame(
            {
                "path": ["a.fits"],
                "x_pix": [300.0],
                "y_pix": [400.0],
                "wcs_ok": [True],
            }
        )
        crop = {"x_min": 0, "y_min": 0, "shape": (10, 10)}
        science = (1.0, 2.0)
        other = (3.0, 4.0)

        class _FakeHDU:
            header = {}

        class _FakeHDUL:
            def __init__(self):
                self._hdu = _FakeHDU()

            def __getitem__(self, idx):
                return self._hdu

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with mock.patch(
            "syndiff_pipeline.common.wcs_grouping.open_fits_memmap",
            return_value=_FakeHDUL(),
        ) as open_fits, mock.patch(
            "astropy.wcs.WCS",
        ) as wcs_cls, mock.patch(
            "syndiff_pipeline.common.wcs_grouping.world_ra_dec_to_pixel",
            return_value=(12.0, 34.0),
        ):
            wcs_cls.return_value = object()
            out = photometry.per_frame_target_crop_xy(
                wcs,
                other[0],
                other[1],
                crop,
                manifest_science_ra_dec=science,
            )
            open_fits.assert_called_once()

        np.testing.assert_allclose(out, [[12.0, 34.0]])


if __name__ == "__main__":
    unittest.main()
