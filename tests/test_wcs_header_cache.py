"""Tests for SCC-scoped WCS header cache paths and dual-write."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.wcs_header_cache import (
    wcs_cache_csv_path,
    wcs_cache_path,
)
from syndiff_pipeline.common.scc_paths import (
    scc_wcs_cache_csv,
    scc_wcs_cache_parquet,
)


class TestWcsHeaderCachePaths(unittest.TestCase):
    def test_parquet_and_csv_under_scc(self):
        with tempfile.TemporaryDirectory() as tmp:
            pq = wcs_cache_path(tmp, 20, 3, 3)
            csv = wcs_cache_csv_path(tmp, 20, 3, 3)
            self.assertEqual(pq, scc_wcs_cache_parquet(tmp, 20, 3, 3))
            self.assertEqual(csv, scc_wcs_cache_csv(tmp, 20, 3, 3))
            self.assertTrue(str(pq).endswith("scc/s0020_c3_k3/wcs_cache.parquet"))
            self.assertTrue(str(csv).endswith("scc/s0020_c3_k3/wcs_cache.csv"))


if __name__ == "__main__":
    unittest.main()
