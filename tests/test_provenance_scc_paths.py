"""Unit tests for provenance-store data_root path helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from syndiff_pipeline.common.scc_paths import (
    provenance_bookkeeping_dir,
    provenance_db_path,
    provenance_spool_dir,
    provenance_spool_file,
    ps1_combined_zarr_dir,
    ps1_combined_zarr_path,
    ps1_convolved_zarr_dir,
    ps1_convolved_zarr_path,
)


class TestProvenanceSccPaths(unittest.TestCase):
    def test_ps1_combined_zarr_paths(self):
        self.assertEqual(
            ps1_combined_zarr_dir("/data"),
            Path("/data/ps1_combined_zarr"),
        )
        self.assertEqual(
            ps1_combined_zarr_path("/data"),
            Path("/data/ps1_combined_zarr/ps1_combined.zarr"),
        )

    def test_ps1_convolved_zarr_paths(self):
        self.assertEqual(
            ps1_convolved_zarr_dir("/data"),
            Path("/data/ps1_convolved_zarr"),
        )
        self.assertEqual(
            ps1_convolved_zarr_path("/data"),
            Path("/data/ps1_convolved_zarr/ps1_convolved.zarr"),
        )

    def test_provenance_bookkeeping_dir(self):
        self.assertEqual(
            provenance_bookkeeping_dir("/data"),
            Path("/data/bookkeeping"),
        )

    def test_provenance_db_path(self):
        self.assertEqual(
            provenance_db_path("/data"),
            Path("/data/bookkeeping/provenance.db"),
        )

    def test_provenance_spool_dir(self):
        self.assertEqual(
            provenance_spool_dir("/data"),
            Path("/data/bookkeeping/spool"),
        )

    def test_provenance_spool_file(self):
        path = provenance_spool_file("/data", "worker01", 1234)
        self.assertEqual(
            path,
            Path("/data/bookkeeping/spool/worker01.1234.jsonl"),
        )
        self.assertEqual(path.name, "worker01.1234.jsonl")


if __name__ == "__main__":
    unittest.main()
