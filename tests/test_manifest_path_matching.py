"""Tests for canonical .fits / .fits.gz manifest path matching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common import wcs_grouping


class TestManifestPathMatching(unittest.TestCase):
    def test_manifest_path_row_index_fits_vs_gz(self):
        wcs = pd.DataFrame(
            {
                "path": ["/data/tess2026039233236-s0100-1-2-0302-s_ffic.fits"],
                "group_id": [3],
            }
        )
        idx = wcs_grouping.manifest_path_row_index(wcs)
        gz = "/data/tess2026039233236-s0100-1-2-0302-s_ffic.fits.gz"
        self.assertEqual(idx[wcs_grouping.canonical_fits_path_key(gz)], 0)
        self.assertEqual(wcs_grouping.ref_manifest_row_index(wcs, gz), 0)

    def test_manifest_path_row_index_gz_manifest_fits_lookup(self):
        wcs = pd.DataFrame(
            {
                "path": ["/data/frame.fits.gz"],
                "group_id": [1],
            }
        )
        self.assertEqual(
            wcs_grouping.ref_manifest_row_index(wcs, "/data/frame.fits"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
