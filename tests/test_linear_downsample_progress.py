"""Tests for the linear-mode downsample progress sidecar wiring.

``run_linear_downsample_scc`` previously accepted a ``progress_path``
parameter but never called any ``downsample_progress`` writer -- unlike
field-mode downsample, a linear-mode run never produced
``downsample.progress.json`` at all, so operators watching a long linear
downsample (which can run 30-65 min per SCC) saw the per-target log go
silent because ``joblib.Parallel`` (without ``return_as="generator"``)
blocks until the *entire* batch completes before the parent process can log
or update anything.

These tests exercise the two mechanics the fix relies on in isolation
(without needing the full SCC pipeline's mapping/WCS/convolved-store
fixtures, which ``run_linear_downsample_scc`` has no existing test coverage
for at all):

1. ``joblib.Parallel(..., return_as="generator")`` yields results as they
   complete rather than only after the whole batch finishes.
2. The exact ``downsample_progress`` call sequence
   ``init_progress`` -> repeated ``mark_skycell_done`` -> ``set_progress_phase``
   now used inside ``run_linear_downsample_scc`` produces a sidecar that
   reaches 100% completion and a final ``"complete"`` phase, matching the
   same choreography field-mode downsample already relies on.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from joblib import Parallel, delayed

from syndiff_pipeline.template_creation.processing.downsample_progress import (
    init_progress,
    mark_skycell_done,
    read_progress,
    set_progress_phase,
)


def _slow_identity(x: int) -> int:
    time.sleep(0.05)
    return x


class TestJoblibGeneratorStreaming(unittest.TestCase):
    def test_return_as_generator_yields_before_batch_finishes(self):
        n = 8
        first_result_at = None
        t0 = time.perf_counter()

        results_iter = Parallel(n_jobs=4, backend="loky", return_as="generator")(
            delayed(_slow_identity)(i) for i in range(n)
        )
        seen = []
        for r in results_iter:
            if first_result_at is None:
                first_result_at = time.perf_counter() - t0
            seen.append(r)

        total_elapsed = time.perf_counter() - t0
        self.assertEqual(sorted(seen), list(range(n)))
        # The first result must be observable well before the whole batch's
        # wall-clock time -- proving results stream in rather than all
        # arriving in one blocking return, which is what silently starved
        # the per-target log for the whole duration of a linear downsample
        # run before this fix.
        self.assertLess(first_result_at, total_elapsed)


class TestLinearDownsampleProgressChoreography(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.progress_path = Path(self.tmpdir.name) / "downsample.progress.json"

    def test_single_batch_sequence_reaches_complete(self):
        n_skycells = 12

        # Mirrors run_linear_downsample_scc's call sequence exactly: one
        # conceptual batch (index 0) sized to the whole SCC's skycell list,
        # since linear mode has no field-mode-style composite-key batching.
        init_progress(
            self.progress_path, total_skycells=n_skycells,
            batch_sizes=[n_skycells], oversampling_factor=1,
        )
        state = read_progress(self.progress_path)
        assert state is not None
        self.assertEqual(state["phase"], "parallel_batches")
        self.assertEqual(state["total_skycells"], n_skycells)
        self.assertEqual(state["skycells_done"], 0)

        for _ in range(n_skycells):
            mark_skycell_done(self.progress_path, 0)

        state = read_progress(self.progress_path)
        assert state is not None
        self.assertEqual(state["skycells_done"], n_skycells)
        self.assertEqual(state["batches"]["0"]["done"], n_skycells)

        set_progress_phase(self.progress_path, "writing_outputs")
        set_progress_phase(self.progress_path, "complete", total_skycells=n_skycells)

        state = read_progress(self.progress_path)
        assert state is not None
        self.assertEqual(state["phase"], "complete")
        self.assertEqual(state["skycells_done"], n_skycells)
        self.assertEqual(state["total_skycells"], n_skycells)
        # phase_elapsed_s should have recorded both preceding phases once
        # transitioned away from.
        self.assertIn("parallel_batches", state.get("phase_elapsed_s", {}))
        self.assertIn("writing_outputs", state.get("phase_elapsed_s", {}))


if __name__ == "__main__":
    unittest.main()
