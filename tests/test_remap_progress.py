"""Tests for remap sidecar progress helpers and syndiff progress display."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.stage_progress import (
    _parse_remap_sidecar,
    read_log_progress,
)
from syndiff_pipeline.template_creation.processing.remap_progress import (
    PROGRESS_FILENAME,
    init_exact_cache,
    init_exact_l4a_cache,
    init_exact_l4b_cache,
    init_progress,
    mark_exact_done,
    mark_exact_l4a_done,
    mark_exact_l4b_done,
    progress_path_for_log,
    read_progress,
    set_progress_phase,
)


class TestRemapProgressSidecar(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def test_progress_path_for_log(self):
        log_path = self.root / "per_target" / "target_a" / "remap.log"
        self.assertEqual(
            progress_path_for_log(log_path),
            self.root / "per_target" / "target_a" / PROGRESS_FILENAME,
        )

    def test_init_and_mark_exact_done(self):
        path = self.root / "remap.progress.json"
        init_progress(path, oversampling_factor=4)
        init_exact_cache(path, 480)
        for _ in range(12):
            mark_exact_done(path)
        state = read_progress(path)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["phase"], "exact_l4a")
        self.assertEqual(state["exact_done"], 12)
        self.assertEqual(state["exact_total"], 480)
        self.assertEqual(state["exact_l4a_done"], 12)
        self.assertEqual(state["exact_l4a_total"], 480)
        self.assertEqual(state["oversampling_factor"], 4)

    def test_mark_exact_done_clamps_to_total(self):
        path = self.root / "remap.progress.json"
        init_exact_cache(path, 2)
        for _ in range(5):
            mark_exact_done(path)
        state = read_progress(path)
        assert state is not None
        self.assertEqual(state["exact_done"], 2)

    def test_set_progress_phase_complete(self):
        path = self.root / "remap.progress.json"
        init_progress(path)
        set_progress_phase(path, "complete", exact_done=480, exact_total=480)
        state = read_progress(path)
        assert state is not None
        self.assertEqual(state["phase"], "complete")
        self.assertEqual(state["exact_done"], 480)
        self.assertEqual(state["exact_total"], 480)

    def test_sidecar_parser_exact_fraction(self):
        log_path = self.root / "per_target" / "t1" / "remap.log"
        log_path.parent.mkdir(parents=True)
        progress_path = progress_path_for_log(log_path)
        init_progress(progress_path, oversampling_factor=4)
        init_exact_cache(progress_path, 480)
        for _ in range(12):
            mark_exact_done(progress_path)

        prog = _parse_remap_sidecar(log_path)
        self.assertIsNotNone(prog)
        assert prog is not None
        self.assertEqual(prog.text, "l4a 12/480 os4")
        self.assertEqual(prog.kind, "fraction")

    def test_sidecar_l4b_fraction(self):
        log_path = self.root / "per_target" / "t1" / "remap.log"
        log_path.parent.mkdir(parents=True)
        progress_path = progress_path_for_log(log_path)
        init_progress(progress_path, oversampling_factor=1)
        init_exact_l4b_cache(progress_path, 73)
        for _ in range(5):
            mark_exact_l4b_done(progress_path)

        prog = _parse_remap_sidecar(log_path)
        self.assertIsNotNone(prog)
        assert prog is not None
        self.assertEqual(prog.text, "l4b 5/73")
        self.assertEqual(prog.kind, "fraction")

    def test_sidecar_complete_shows_full_fraction(self):
        log_path = self.root / "per_target" / "t1" / "remap.log"
        log_path.parent.mkdir(parents=True)
        progress_path = progress_path_for_log(log_path)
        set_progress_phase(progress_path, "complete", exact_done=480, exact_total=480)

        prog = read_log_progress(log_path, "remap")
        self.assertIsNotNone(prog)
        assert prog is not None
        self.assertEqual(prog.text, "l4a 480/480")
        self.assertEqual(prog.kind, "fraction")

    def test_sidecar_grouping_phase(self):
        log_path = self.root / "per_target" / "t1" / "remap.log"
        log_path.parent.mkdir(parents=True)
        set_progress_phase(progress_path_for_log(log_path), "grouping")

        prog = read_log_progress(log_path, "remap")
        self.assertIsNotNone(prog)
        assert prog is not None
        self.assertEqual(prog.text, "grouping")
        self.assertEqual(prog.kind, "phase")

    def test_no_fcntl_flock_in_module(self):
        import inspect

        import syndiff_pipeline.template_creation.processing.remap_progress as mod

        src = inspect.getsource(mod)
        self.assertNotIn("import fcntl", src)
        self.assertNotIn("fcntl.flock", src)

    def test_parent_callback_increments_like_parallel_drain(self):
        """Simulate parent on_result drain (hotpants pattern)."""
        path = self.root / "remap.progress.json"
        init_progress(path)
        init_exact_l4a_cache(path, 5)
        statuses = ["write", "skip", "write", "fail", "skip"]
        for _status in statuses:
            mark_exact_l4a_done(path)
        state = read_progress(path)
        assert state is not None
        self.assertEqual(state["exact_l4a_done"], 5)
        self.assertEqual(state["exact_l4a_total"], 5)


if __name__ == "__main__":
    unittest.main()
