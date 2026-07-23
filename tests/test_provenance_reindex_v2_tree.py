"""Tests for v2 SCC diff-lane coverage in reindex and gc."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.provenance.gc import gc_report
from syndiff_pipeline.common.provenance.reindex import reindex_scc_tree
from syndiff_pipeline.common.provenance.store import ProvenanceStore


class TestReindexV2DiffTree(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_root = Path(self._tmp.name)
        self.scc_dir = self.data_root / "s0020" / "c1" / "k1"
        self.store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")

    @staticmethod
    def _make_label_dir(
        diff_root: Path,
        workspace_label: str,
        *,
        filename: str = "tess2020019142923_hp_d.fits.fz",
    ) -> Path:
        label_dir = diff_root / workspace_label
        label_dir.mkdir(parents=True)
        (label_dir / filename).write_bytes(b"fake")
        return label_dir

    def test_reindex_registers_default_and_named_diff_lanes(self) -> None:
        default_label = self._make_label_dir(self.scc_dir / "diff", "hp_d")
        named_label = self._make_label_dir(self.scc_dir / "diff_somelane", "hp_d")

        n = reindex_scc_tree(self.store, self.scc_dir, 20, 1, 1)
        self.assertEqual(n, 2)

        default_rows = self.store.artifacts_by_kind_spatial(
            "diff_image_legacy_unverified",
            {
                "s": 20,
                "c": 1,
                "k": 1,
                "workspace_label": "hp_d",
            },
        )
        self.assertEqual(len(default_rows), 1)
        self.assertEqual(default_rows[0].location, str(default_label))

        named_rows = self.store.artifacts_by_kind_spatial(
            "diff_image_legacy_unverified",
            {
                "s": 20,
                "c": 1,
                "k": 1,
                "workspace_label": "hp_d",
                "store_name": "somelane",
            },
        )
        self.assertEqual(len(named_rows), 1)
        self.assertEqual(named_rows[0].location, str(named_label))

    def test_gc_report_lists_diff_recipe_dirs_in_all_lanes(self) -> None:
        default_label = self._make_label_dir(self.scc_dir / "diff", "hp_d")
        named_label = self._make_label_dir(self.scc_dir / "diff_somelane", "ks_d")
        # Event subtree under a diff lane must not be counted as a label dir.
        (self.scc_dir / "diff" / "events" / "evt1").mkdir(parents=True)

        report = gc_report(self.data_root)
        self.assertEqual(len(report.diff_recipe_dirs), 2)
        self.assertIn(str(default_label), report.diff_recipe_dirs)
        self.assertIn(str(named_label), report.diff_recipe_dirs)

    def test_reindex_skips_empty_label_dirs(self) -> None:
        empty = self.scc_dir / "diff" / "hp_d"
        empty.mkdir(parents=True)
        self._make_label_dir(self.scc_dir / "diff", "hp_d_has_content")

        n = reindex_scc_tree(self.store, self.scc_dir, 20, 1, 1)
        self.assertEqual(n, 1)

    def test_reindex_infers_background_and_shared_mask_kinds(self) -> None:
        bkg_label = self._make_label_dir(
            self.scc_dir / "diff",
            "hp_bkg",
            filename="tess2020019142923_hp_bkg.fits.fz",
        )
        mask_label = self._make_label_dir(
            self.scc_dir / "diff",
            "shared_mask",
            filename="shared_mask.fits.fz",
        )

        n = reindex_scc_tree(self.store, self.scc_dir, 20, 1, 1)
        self.assertEqual(n, 2)

        bkg_rows = self.store.artifacts_by_kind_spatial(
            "diff_background_legacy_unverified",
            {
                "s": 20,
                "c": 1,
                "k": 1,
                "workspace_label": "hp_bkg",
            },
        )
        self.assertEqual(len(bkg_rows), 1)
        self.assertEqual(bkg_rows[0].location, str(bkg_label))

        mask_rows = self.store.artifacts_by_kind_spatial(
            "shared_mask_legacy_unverified",
            {"s": 20, "c": 1, "k": 1},
        )
        self.assertEqual(len(mask_rows), 1)
        self.assertEqual(mask_rows[0].location, str(mask_label))


if __name__ == "__main__":
    unittest.main()
