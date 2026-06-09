"""Integration tests for downsample sidecar progress under concurrency and lifecycle."""
from __future__ import annotations

import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from syndiff_pipeline.template.downsample import process_skycell_batch
from syndiff_pipeline.template.downsample_progress import (
    init_progress,
    mark_skycell_done,
    progress_path_for_log,
    read_progress,
    set_progress_phase,
)
from syndiff_pipeline.template_runner.stage_progress import read_log_progress


def _worker_mark(path_str: str, batch_idx: int, count: int) -> None:
    path = Path(path_str)
    for _ in range(count):
        mark_skycell_done(path, batch_idx)


class TestDownsampleProgressIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def test_concurrent_workers_reach_exact_total(self):
        path = self.root / "downsample.progress.json"
        batch_sizes = [7, 7, 7, 6]
        init_progress(path, total_skycells=sum(batch_sizes), batch_sizes=batch_sizes)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=4) as pool:
            pool.starmap(
                _worker_mark,
                [(str(path), i, batch_sizes[i]) for i in range(len(batch_sizes))],
            )

        state = read_progress(path)
        assert state is not None
        self.assertEqual(state["skycells_done"], 27)
        self.assertEqual(state["total_skycells"], 27)
        for i, size in enumerate(batch_sizes):
            self.assertEqual(state["batches"][str(i)]["done"], size)

    def test_full_lifecycle_phases(self):
        log_path = self.root / "per_target" / "t1" / "downsample.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text("Precomputing shifts for all offsets...\n", encoding="utf-8")
        path = progress_path_for_log(log_path)

        set_progress_phase(path, "precomputing_shifts", offsets_done=0, offsets_total=4)
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "shifts 0/4")

        set_progress_phase(path, "precomputing_shifts", offsets_done=2, offsets_total=4)
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "shifts 2/4")

        init_progress(path, total_skycells=5, batch_sizes=[3, 2])
        mark_skycell_done(path, 0)
        mark_skycell_done(path, 0)
        mark_skycell_done(path, 1)
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "3/5")

        set_progress_phase(path, "combining", total_skycells=5)
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "combining")

        set_progress_phase(path, "complete", total_skycells=5)
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "5/5")

    def test_sidecar_overrides_misleading_log(self):
        log_path = self.root / "downsample.log"
        log_path.write_text(
            "\n".join(
                [
                    "Processing 84 skycells in 12 batches...",
                    "Completed batch 12",
                    "Completed batch 11",
                ]
            ),
            encoding="utf-8",
        )
        sidecar = progress_path_for_log(log_path)
        sidecar.write_text(
            '{"total_skycells": 84, "skycells_done": 10, "phase": "parallel_batches"}\n',
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "10/84")

    def test_sidecar_without_log_file(self):
        log_path = self.root / "missing" / "downsample.log"
        sidecar = progress_path_for_log(log_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(
            '{"total_skycells": 20, "skycells_done": 7, "phase": "parallel_batches"}\n',
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "7/20")

    def test_process_skycell_batch_marks_each_skycell_on_failure(self):
        path = self.root / "downsample.progress.json"
        init_progress(path, total_skycells=2, batch_sizes=[2])

        offsets = np.array([[0.0, 0.0]])
        shifts_dict = {(0.0, 0.0): __import__("pandas").DataFrame(
            {"NAME": ["a", "b"], "shift_x": [0, 0], "shift_y": [0, 0]}
        )}

        with mock.patch("syndiff_pipeline.template.downsample.fits.open") as mock_open, mock.patch(
            "syndiff_pipeline.template.downsample.zarr.open"
        ) as mock_zarr:
            mock_open.side_effect = OSError("registration read failed")
            mock_zarr.return_value = {}
            process_skycell_batch(
                0,
                ["reg1.fits", "reg2.fits"],
                ["skycell.a", "skycell.b"],
                offsets,
                shifts_dict,
                (10, 10),
                Path("/nonexistent.zarr"),
                (0, 0, 5, 5),
                progress_path=path,
            )

        state = read_progress(path)
        assert state is not None
        self.assertEqual(state["skycells_done"], 2)
        self.assertEqual(state["batches"]["0"]["done"], 2)


if __name__ == "__main__":
    unittest.main()
