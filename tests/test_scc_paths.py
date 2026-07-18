"""Unit tests for nested SCC data_root path helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    scc_label,
    scc_root,
    scc_templates_dir,
)


class TestSccPaths(unittest.TestCase):
    def test_scc_root_nested(self):
        root = scc_root("/data", 15, 1, 1)
        self.assertEqual(root, Path("/data/s0015/c1/k1"))

    def test_scc_templates_dir_nested(self):
        path = scc_templates_dir("/data", 20, 3, 3, oversampling_factor=1)
        self.assertEqual(path, Path("/data/s0020/c3/k3/templates/oversampling_1"))

    def test_scc_label_unchanged(self):
        self.assertEqual(scc_label(15, 1, 1), "s0015_c1_k1")

    def test_event_scc_leaf_still_flat_label(self):
        leaf = event_scc_leaf("/ws", "2020ftl", 15, 1, 1)
        self.assertEqual(leaf, Path("/ws/events/2020ftl/s0015_c1_k1"))


if __name__ == "__main__":
    unittest.main()
