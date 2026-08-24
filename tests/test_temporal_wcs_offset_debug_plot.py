"""Tests for the temporal_wcs vs native-WCS offset debug plot.

Both position sources read from cached data only (no FITS opens): the
temporal_wcs model store and the SCC's ``ffi_list`` header cache.
"""

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
    """Covered stems "a" and "b" with distinct temporal_wcs positions."""

    def __init__(self, root):
        self.root = root
        self.frame_contract = {"model_origin_ffi": [0, 0]}
        self.frames = pd.DataFrame({"btjd": [100.0, 101.0]}, index=["a", "b"])

    def raw_for_stem(self, stem):
        if stem == "a":
            return _FakeModel(11.0, 21.0), 100.0
        if stem == "b":
            return _FakeModel(9.0, 19.0), 101.0
        raise KeyError(stem)


class _FakeWcs:
    def __init__(self, xy):
        self.xy = xy


def _fake_world_to_pixel(wcs, ra, dec):
    return wcs.xy


class TestTemporalWcsOffsetDebugPlot(unittest.TestCase):
    def _patches(self, wcs_from_row):
        return (
            mock.patch(
                "syndiff_pipeline.difference_imaging.wcs.temporal_cheb.TemporalChebWcsStore",
                side_effect=_FakeStore,
            ),
            mock.patch(
                "syndiff_pipeline.common.download.manifest_basename_from_local",
                side_effect=lambda p: str(p).replace(".fits", ""),
            ),
            mock.patch(
                "syndiff_pipeline.common.wcs_header_cache.wcs_from_cached_row",
                side_effect=wcs_from_row,
            ),
            mock.patch(
                "syndiff_pipeline.common.wcs_grouping.world_ra_dec_to_pixel",
                side_effect=_fake_world_to_pixel,
            ),
        )

    def test_writes_png_using_only_wcs_ok_and_temporal_wcs_covered_frames(self):
        # "a" and "b": wcs_ok=True, covered by temporal_wcs -> compared.
        # "c": wcs_ok=False -> must be excluded even though temporal_wcs covers "c"-like stems.
        # "missing": wcs_ok=True but not covered by temporal_wcs -> excluded.
        ffi_list_df = pd.DataFrame(
            {
                "wcs_ok": [True, True, False, True],
                "_native_xy": [(10.0, 20.0), (10.0, 20.0), (10.0, 20.0), (10.0, 20.0)],
            },
            index=pd.Index(["a", "b", "b_bad", "missing"], name="filename"),
        )

        def _wcs_from_row(row):
            return _FakeWcs(row["_native_xy"])

        wcs_table = pd.DataFrame(
            {"path": ["a.fits", "b.fits", "b_bad.fits", "missing.fits"]}
        )
        crop = {"x_min": 0, "y_min": 0, "shape": (10, 10)}

        patches = self._patches(_wcs_from_row)
        with patches[0], patches[1], patches[2], patches[3]:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "offset.png"
                result = photometry.write_temporal_wcs_offset_debug_plot(
                    wcs_table,
                    1.0,
                    2.0,
                    crop,
                    "/fake/wcs/dir",
                    ffi_list_df,
                    png_path=str(out),
                )
                self.assertEqual(result, str(out))
                self.assertTrue(out.is_file())
                self.assertGreater(out.stat().st_size, 0)

    def test_no_usable_frames_returns_none(self):
        ffi_list_df = pd.DataFrame(
            {"wcs_ok": [False], "_native_xy": [(10.0, 20.0)]},
            index=pd.Index(["a"], name="filename"),
        )
        wcs_table = pd.DataFrame({"path": ["a.fits"]})
        crop = {"x_min": 0, "y_min": 0, "shape": (10, 10)}

        def _wcs_from_row(row):
            return _FakeWcs(row["_native_xy"])

        patches = self._patches(_wcs_from_row)
        with patches[0], patches[1], patches[2], patches[3]:
            result = photometry.write_temporal_wcs_offset_debug_plot(
                wcs_table,
                1.0,
                2.0,
                crop,
                "/fake/wcs/dir",
                ffi_list_df,
                png_path="/tmp/should_not_be_written.png",
            )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
