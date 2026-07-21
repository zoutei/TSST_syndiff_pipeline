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
    def _make_recipe_dir(
        diff_root: Path,
        workspace_label: str,
        recipe_fp: str,
        *,
        filename: str = "tess2020019142923_hp_d.fits.fz",
    ) -> Path:
        recipe_dir = diff_root / workspace_label / recipe_fp
        recipe_dir.mkdir(parents=True)
        (recipe_dir / filename).write_bytes(b"fake")
        return recipe_dir

    def test_reindex_registers_default_and_named_diff_lanes(self) -> None:
        default_recipe = self._make_recipe_dir(self.scc_dir / "diff", "hp_d", "recipefp1")
        named_recipe = self._make_recipe_dir(
            self.scc_dir / "diff_somelane", "hp_d", "recipefp2"
        )

        n = reindex_scc_tree(self.store, self.scc_dir, 20, 1, 1)
        self.assertEqual(n, 2)

        default_rows = self.store.artifacts_by_kind_spatial(
            "diff_image_legacy_unverified",
            {
                "s": 20,
                "c": 1,
                "k": 1,
                "workspace_label": "hp_d",
                "recipe_fp": "recipefp1",
            },
        )
        self.assertEqual(len(default_rows), 1)
        self.assertEqual(default_rows[0].location, str(default_recipe))

        named_rows = self.store.artifacts_by_kind_spatial(
            "diff_image_legacy_unverified",
            {
                "s": 20,
                "c": 1,
                "k": 1,
                "workspace_label": "hp_d",
                "recipe_fp": "recipefp2",
                "store_name": "somelane",
            },
        )
        self.assertEqual(len(named_rows), 1)
        self.assertEqual(named_rows[0].location, str(named_recipe))

    def test_gc_report_lists_diff_recipe_dirs_in_all_lanes(self) -> None:
        default_recipe = self._make_recipe_dir(self.scc_dir / "diff", "hp_d", "recipefp1")
        named_recipe = self._make_recipe_dir(
            self.scc_dir / "diff_somelane", "ks_d", "recipefp2"
        )
        # Event subtree under a diff lane must not be counted as a recipe dir.
        (self.scc_dir / "diff" / "events" / "evt1").mkdir(parents=True)

        report = gc_report(self.data_root)
        self.assertEqual(len(report.diff_recipe_dirs), 2)
        self.assertIn(str(default_recipe), report.diff_recipe_dirs)
        self.assertIn(str(named_recipe), report.diff_recipe_dirs)

    def test_reindex_skips_empty_recipe_dirs(self) -> None:
        empty = self.scc_dir / "diff" / "hp_d" / "empty_recipe"
        empty.mkdir(parents=True)
        self._make_recipe_dir(self.scc_dir / "diff", "hp_d", "has_content")

        n = reindex_scc_tree(self.store, self.scc_dir, 20, 1, 1)
        self.assertEqual(n, 1)

    def test_reindex_infers_background_and_shared_mask_kinds(self) -> None:
        bkg_recipe = self._make_recipe_dir(
            self.scc_dir / "diff",
            "hp_bkg",
            "bkgfp",
            filename="tess2020019142923_hp_bkg.fits.fz",
        )
        mask_recipe = self._make_recipe_dir(
            self.scc_dir / "diff",
            "shared_mask",
            "maskfp",
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
                "recipe_fp": "bkgfp",
            },
        )
        self.assertEqual(len(bkg_rows), 1)
        self.assertEqual(bkg_rows[0].location, str(bkg_recipe))

        mask_rows = self.store.artifacts_by_kind_spatial(
            "shared_mask_legacy_unverified",
            {"s": 20, "c": 1, "k": 1},
        )
        self.assertEqual(len(mask_rows), 1)
        self.assertEqual(mask_rows[0].location, str(mask_recipe))


if __name__ == "__main__":
    unittest.main()
