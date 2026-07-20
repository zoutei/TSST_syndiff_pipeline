"""Unit tests for gid_epoch_index.npz → dict loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing.field_remap import load_gid_epoch_index


def _write_mini_index(path: Path, *, n_l4a: int = 3, n_l4b: int = 2) -> None:
    skycells = np.array([f"skycell.{i}.1" for i in range(n_l4a)], dtype=object)
    np.savez_compressed(
        path,
        l4a_skycell=skycells,
        l4a_gid=np.arange(n_l4a, dtype=np.int32),
        l4a_sx=np.zeros(n_l4a, dtype=np.int32),
        l4a_sy=np.arange(n_l4a, dtype=np.int32),
        l4a_epoch_id=np.arange(100, 100 + n_l4a, dtype=np.int32),
        l4b_pair_lo=np.zeros(n_l4b, dtype=np.int32),
        l4b_pair_hi=np.ones(n_l4b, dtype=np.int32),
        l4b_gid=np.arange(n_l4b, dtype=np.int32),
        l4b_sx_lo=np.zeros(n_l4b, dtype=np.int32),
        l4b_sy_lo=np.zeros(n_l4b, dtype=np.int32),
        l4b_sx_hi=np.ones(n_l4b, dtype=np.int32),
        l4b_sy_hi=np.zeros(n_l4b, dtype=np.int32),
        l4b_pair_epoch_id=np.arange(200, 200 + n_l4b, dtype=np.int32),
    )


class TestLoadGidEpochIndex(unittest.TestCase):
    def test_loads_both_halves(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gid_epoch_index.npz"
            _write_mini_index(path)
            idx = load_gid_epoch_index(path)
            self.assertEqual(len(idx["l4a"]), 3)
            self.assertEqual(len(idx["l4b"]), 2)
            self.assertEqual(idx["l4a"][("skycell.1.1", 1, 0, 1)], 101)
            self.assertEqual(idx["l4b"][(0, 1, 1, 0, 0, 1, 0)], 201)

    def test_include_inter_false_skips_l4b(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gid_epoch_index.npz"
            _write_mini_index(path, n_l4b=5)
            idx = load_gid_epoch_index(path, include_inter=False)
            self.assertEqual(len(idx["l4a"]), 3)
            self.assertEqual(idx["l4b"], {})


if __name__ == "__main__":
    unittest.main()
