"""Tests for the temporal_wcs-backed forced-target position resolver."""

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


class _FakeModel:
    def __init__(self, x, y):
        self._xy = (x, y)

    def world_to_pixel_values(self, ra, dec, btjd):
        return self._xy


class _FakeStore:
    """Mimics TemporalChebWcsStore for two stems, one missing from the fit."""

    def __init__(self, root):
        self.root = root
        self.frame_contract = {"model_origin_ffi": [0, 0]}

    def raw_for_stem(self, stem):
        if stem == "missing":
            raise KeyError(stem)
        return _FakeModel(50.0, 60.0), 2600.0


class TestPerFrameTargetCropXYTemporalWcs(unittest.TestCase):
    def test_uses_temporal_model_and_falls_back_for_missing_stem(self):
        wcs = pd.DataFrame({"path": ["present.fits", "missing.fits"]})
        crop = {"x_min": 0, "y_min": 0, "shape": (10, 10)}

        def _stem(p):
            return "missing" if "missing" in str(p) else "present"

        class _FakeHDU:
            header = {}

        class _FakeHDUL:
            def __getitem__(self, idx):
                return _FakeHDU()

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with mock.patch(
            "syndiff_pipeline.difference_imaging.wcs.temporal_cheb.TemporalChebWcsStore",
            side_effect=_FakeStore,
        ), mock.patch(
            "syndiff_pipeline.common.download.manifest_basename_from_local",
            side_effect=_stem,
        ), mock.patch(
            "syndiff_pipeline.common.wcs_grouping.open_fits_memmap",
            return_value=_FakeHDUL(),
        ), mock.patch(
            "astropy.wcs.WCS", return_value=object(),
        ), mock.patch(
            "syndiff_pipeline.common.wcs_grouping.world_ra_dec_to_pixel",
            return_value=(7.0, 8.0),
        ):
            out = photometry.per_frame_target_crop_xy_temporal_wcs(
                wcs, 10.0, 20.0, crop, "/fake/wcs/dir",
            )

        np.testing.assert_allclose(out, [[50.0, 60.0], [7.0, 8.0]])

    def test_origin_mismatch_raises(self):
        wcs = pd.DataFrame({"path": ["present.fits"]})
        crop = {"x_min": 44, "y_min": 0, "shape": (10, 10)}

        with mock.patch(
            "syndiff_pipeline.difference_imaging.wcs.temporal_cheb.TemporalChebWcsStore",
            side_effect=_FakeStore,
        ):
            with self.assertRaises(ValueError) as ctx:
                photometry.per_frame_target_crop_xy_temporal_wcs(
                    wcs, 10.0, 20.0, crop, "/fake/wcs/dir",
                )
        self.assertIn("model_origin_ffi", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
