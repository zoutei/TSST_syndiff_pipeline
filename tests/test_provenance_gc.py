"""Tests for report-only provenance GC."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.provenance.gc import gc_report
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path, ps1_combined_zarr_path


class TestProvenanceGc(unittest.TestCase):
    def test_gc_report_empty_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = gc_report(tmp)
            self.assertEqual(report.db_artifacts, 0)
            self.assertEqual(report.orphan_fingerprint_dirs, [])

    def test_gc_report_flags_orphan_fingerprint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            orphan = ps1_combined_zarr_path(data) / "STAPS" / "1234N567" / "orphanfp123"
            orphan.mkdir(parents=True)
            (orphan / "_provenance.json").write_text("{}", encoding="utf-8")

            ProvenanceStore(provenance_db_path(data))

            report = gc_report(data)
            self.assertEqual(len(report.orphan_fingerprint_dirs), 1)
            self.assertIn("orphanfp123", report.orphan_fingerprint_dirs[0])


if __name__ == "__main__":
    unittest.main()
