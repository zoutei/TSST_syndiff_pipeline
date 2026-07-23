"""Unit tests for nested SCC data_root path helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    normalize_store_name,
    ps1_combined_zarr_path,
    ps1_convolved_zarr_path,
    ps1_skycells_zarr_dir,
    ps1_skycells_zarr_lock_path,
    ps1_skycells_zarr_path,
    provenance_db_path,
    provenance_spool_dir,
    scc_debug_plots_dir,
    scc_diff_label_dir,
    scc_diff_workspace_dir,
    scc_label,
    scc_remap_dir,
    scc_root,
    scc_templates_dir,
    store_subdir,
)


class TestSccPaths(unittest.TestCase):
    def test_scc_root_nested(self):
        root = scc_root("/data", 15, 1, 1)
        self.assertEqual(root, Path("/data/s0015/c1/k1"))

    def test_scc_templates_dir_nested(self):
        path = scc_templates_dir("/data", 20, 3, 3, oversampling_factor=1)
        self.assertEqual(path, Path("/data/s0020/c3/k3/templates/oversampling_1"))

    def test_scc_remap_dir_nested(self):
        path = scc_remap_dir("/data", 20, 3, 3, oversampling_factor=2)
        self.assertEqual(path, Path("/data/s0020/c3/k3/remap/oversampling_2"))

    def test_scc_label_unchanged(self):
        self.assertEqual(scc_label(15, 1, 1), "s0015_c1_k1")

    def test_event_scc_leaf_still_flat_label(self):
        leaf = event_scc_leaf("/ws", "2020ftl", 15, 1, 1)
        self.assertEqual(leaf, Path("/ws/events/2020ftl/s0015_c1_k1"))

    def test_ps1_skycells_zarr_paths(self):
        self.assertEqual(
            ps1_skycells_zarr_dir("/data"),
            Path("/data/ps1_skycells_zarr"),
        )
        self.assertEqual(
            ps1_skycells_zarr_path("/data"),
            Path("/data/ps1_skycells_zarr/ps1_skycells.zarr"),
        )
        self.assertEqual(
            ps1_skycells_zarr_lock_path("/data"),
            Path("/data/ps1_skycells_zarr/ps1_skycells.zarr.lock"),
        )

    def test_ps1_combined_and_convolved_zarr_paths_share_ps1_skycells_zarr_dir(self):
        # Provenance plan decision #14: all three PS1 stores live under the
        # same ps1_skycells_zarr/ directory.
        self.assertEqual(
            ps1_combined_zarr_path("/data"),
            Path("/data/ps1_skycells_zarr/ps1_combined.zarr"),
        )
        self.assertEqual(
            ps1_convolved_zarr_path("/data"),
            Path("/data/ps1_skycells_zarr/ps1_convolved.zarr"),
        )
        self.assertEqual(
            ps1_combined_zarr_path("/data").parent, ps1_skycells_zarr_dir("/data")
        )
        self.assertEqual(
            ps1_convolved_zarr_path("/data").parent, ps1_skycells_zarr_dir("/data")
        )

    def test_provenance_db_and_spool_paths_are_data_root_scoped(self):
        self.assertEqual(
            provenance_db_path("/data"), Path("/data/bookkeeping/provenance.db")
        )
        self.assertEqual(
            provenance_spool_dir("/data"), Path("/data/bookkeeping/spool")
        )

    def test_scc_diff_store_paths(self):
        label_dir = scc_diff_label_dir(
            "/data", 20, 3, 3, store_name="lane_a", label="hp_d"
        )
        self.assertEqual(
            label_dir,
            Path("/data/s0020/c3/k3/diff_lane_a/hp_d"),
        )
        self.assertEqual(
            scc_diff_workspace_dir(
                "/data", 20, 3, 3, store_name=None, workspace_label="hp_d"
            ),
            Path("/data/s0020/c3/k3/diff/hp_d"),
        )

    def test_named_store_subdir_and_paths(self):
        self.assertEqual(store_subdir("templates", None), "templates")
        self.assertEqual(store_subdir("templates", "no_l4b"), "templates_no_l4b")
        self.assertEqual(store_subdir("remap", "hybrid_r2"), "remap_hybrid_r2")
        self.assertEqual(normalize_store_name("  "), None)
        self.assertEqual(normalize_store_name("no_l4b"), "no_l4b")
        with self.assertRaises(ValueError):
            normalize_store_name("../evil")
        with self.assertRaises(ValueError):
            normalize_store_name("a/b")

        self.assertEqual(
            scc_templates_dir(
                "/data", 20, 1, 1, oversampling_factor=2, store_name="no_l4b"
            ),
            Path("/data/s0020/c1/k1/templates_no_l4b/oversampling_2"),
        )
        self.assertEqual(
            scc_remap_dir(
                "/data", 20, 1, 1, oversampling_factor=1, store_name="hybrid_r2"
            ),
            Path("/data/s0020/c1/k1/remap_hybrid_r2/oversampling_1"),
        )
        self.assertEqual(
            scc_debug_plots_dir("/data", 20, 1, 1),
            Path("/data/s0020/c1/k1/debug_plots"),
        )


if __name__ == "__main__":
    unittest.main()
