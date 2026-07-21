"""Unit tests for gid_epoch_index.npz → dict loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing.field_remap import (
    _write_gid_epoch_index,
    load_gid_epoch_index,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    FRAME_ORIGIN_MEASURED,
    ShiftSchedule,
    assign_groups_from_schedule,
    build_shift_epochs,
)


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


class TestWriteGidEpochIndex(unittest.TestCase):
    def test_write_from_shift_epochs_and_members(self):
        skycell_names = np.array(["c0"])
        sx = np.array([[1], [1], [0], [1]], dtype=np.int16)
        sy = np.zeros_like(sx)
        schedule = ShiftSchedule(
            skycell_names=skycell_names,
            sx_float=sx.astype(np.float32),
            sy_float=sy.astype(np.float32),
            sx_int=sx,
            sy_int=sy,
            frame_valid=np.ones(4, dtype=bool),
            frame_origin=np.full(4, FRAME_ORIGIN_MEASURED, dtype=np.int8),
            meta={"schema_version": 1},
        )
        assignment = assign_groups_from_schedule(
            schedule,
            grouping_quantum_ps1_px=1.0,
            cache_quantum_ps1_px=0.25,
            keying="phase",
        )
        shift_epochs, members = build_shift_epochs(
            schedule, assignment.group_id_per_frame
        )
        pair_epochs = pd.DataFrame(
            {
                "pair_epoch_id": pd.Series(dtype="int32"),
                "id_lo": pd.Series(dtype="int32"),
                "id_hi": pd.Series(dtype="int32"),
                "sx_lo": pd.Series(dtype="int32"),
                "sy_lo": pd.Series(dtype="int32"),
                "sx_hi": pd.Series(dtype="int32"),
                "sy_hi": pd.Series(dtype="int32"),
                "frame_lo": pd.Series(dtype="int32"),
                "frame_hi": pd.Series(dtype="int32"),
                "n_frames": pd.Series(dtype="int32"),
                "n_measured_frames": pd.Series(dtype="int32"),
                "gid_begin": pd.Series(dtype="int32"),
                "gid_end": pd.Series(dtype="int32"),
                "n_groups": pd.Series(dtype="int32"),
                "rep_frame_index": pd.Series(dtype="int32"),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gid_epoch_index.npz"
            _write_gid_epoch_index(
                path,
                shift_epochs=shift_epochs,
                pair_epochs=pair_epochs,
                members=members,
            )
            idx = load_gid_epoch_index(path)
            self.assertEqual(len(idx["l4a"]), len(members))
            self.assertEqual(idx["l4b"], {})
            for row in members.itertuples(index=False):
                key = (str(row.scope_key), int(row.group_id), 1, 0)
                self.assertIn(key, idx["l4a"])
                self.assertEqual(int(idx["l4a"][key]), int(row.epoch_id))


if __name__ == "__main__":
    unittest.main()
