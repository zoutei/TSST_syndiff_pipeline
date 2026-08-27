"""Regression tests: SCC field-lane FFI selection must filter wcs_ok even
when uncapped (no max_ffis), not just under the smoke-test max_ffis path.

Root cause of a real production failure (2026-08-22): an uncapped run
processed every FFI on disk including manifest rows with wcs_ok=False
(group_id=-1, NaN group_dx/group_dy), which crashed downstream in
background_estimate instead of being skipped.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.execute import (
    _ffi_paths_for_processing,
    _select_ffis_for_field_lane,
)
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig


def _wcs_table():
    return pd.DataFrame(
        {
            "path": ["/a/f1.fits", "/a/f2.fits", "/a/f3.fits"],
            "wcs_ok": [True, False, True],
            "group_id": [0, -1, 1],
            "group_dx": [0.0, float("nan"), 0.01],
            "group_dy": [0.0, float("nan"), 0.01],
        }
    )


class TestSelectFfisForFieldLane(unittest.TestCase):
    def test_uncapped_filters_wcs_ok_false(self):
        selected = _select_ffis_for_field_lane(
            ["/a/f1.fits", "/a/f2.fits", "/a/f3.fits"],
            _wcs_table(),
            max_ffis=None,
        )
        self.assertEqual(selected, ["/a/f1.fits", "/a/f3.fits"])

    def test_capped_filters_wcs_ok_false(self):
        selected = _select_ffis_for_field_lane(
            ["/a/f1.fits", "/a/f2.fits", "/a/f3.fits"],
            _wcs_table(),
            max_ffis=2,
        )
        self.assertEqual(selected, ["/a/f1.fits", "/a/f3.fits"])

    def test_uncapped_no_wcs_table_returns_all(self):
        selected = _select_ffis_for_field_lane(
            ["/a/f1.fits", "/a/f2.fits"], None, max_ffis=None
        )
        self.assertEqual(selected, ["/a/f1.fits", "/a/f2.fits"])


class TestFfiPathsForProcessingFieldLane(unittest.TestCase):
    def test_uncapped_field_lane_excludes_wcs_ok_false(self):
        cfg = SynDiffConfig()
        cfg.sector, cfg.camera, cfg.ccd = 20, 3, 3
        cfg.target_ra = None
        cfg.target_dec = None
        cfg.max_ffis = None
        with mock.patch(
            "syndiff_pipeline.difference_imaging.orchestration.execute._sorted_local_ffis",
            return_value=["/a/f1.fits", "/a/f2.fits", "/a/f3.fits"],
        ):
            selected = _ffi_paths_for_processing(cfg, wcs_table=_wcs_table())
        self.assertEqual(selected, ["/a/f1.fits", "/a/f3.fits"])


if __name__ == "__main__":
    unittest.main()
