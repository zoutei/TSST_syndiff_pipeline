"""Tests for downsample sidecar progress helpers."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template.downsample_progress import (
    init_progress,
    mark_skycell_done,
    progress_path_for_log,
    read_progress,
    set_progress_phase,
)


class TestDownsampleProgressSidecar(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def test_progress_path_for_log(self):
        log_path = self.root / "per_target" / "target_a" / "downsample.log"
        self.assertEqual(
            progress_path_for_log(log_path),
            self.root / "per_target" / "target_a" / "downsample.progress.json",
        )

    def test_init_and_mark_skycell_done(self):
        path = self.root / "downsample.progress.json"
        init_progress(path, total_skycells=10, batch_sizes=[4, 3, 3])

        mark_skycell_done(path, 0)
        mark_skycell_done(path, 0)
        mark_skycell_done(path, 1)
        state = read_progress(path)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["skycells_done"], 3)
        self.assertEqual(state["batches"]["0"]["done"], 2)
        self.assertEqual(state["batches"]["1"]["done"], 1)

    def test_mark_skycell_done_clamps_to_batch_size(self):
        path = self.root / "downsample.progress.json"
        init_progress(path, total_skycells=2, batch_sizes=[2])
        for _ in range(5):
            mark_skycell_done(path, 0)
        state = read_progress(path)
        assert state is not None
        self.assertEqual(state["batches"]["0"]["done"], 2)
        self.assertEqual(state["skycells_done"], 2)

    def test_set_progress_phase(self):
        path = self.root / "downsample.progress.json"
        set_progress_phase(path, "precomputing_shifts", offsets_done=2, offsets_total=5)
        state = read_progress(path)
        assert state is not None
        self.assertEqual(state["phase"], "precomputing_shifts")
        self.assertEqual(state["offsets_done"], 2)
        self.assertEqual(state["offsets_total"], 5)


if __name__ == "__main__":
    unittest.main()
