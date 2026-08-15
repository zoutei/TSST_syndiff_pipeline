"""Regression test for _cleanup_child_processes' executor-shutdown race.

Background: ``ProcessPoolExecutor.shutdown()`` can itself null out the
executor's internal ``_processes`` dict as part of its own teardown. When
``_cleanup_child_processes`` called ``.shutdown()`` and then immediately
iterated ``getattr(executor, "_processes", {}).values()``, the attribute
*existed* (as ``None``) by that point, so getattr's ``{}`` default never
applied and ``.values()`` raised ``AttributeError: 'NoneType' object has no
attribute 'values'``. Observed in production: a real CVZ ps1_process run
(s0055) completed all of its actual work ("Pipeline completed
successfully!") and then exited 1 from this crash in its `finally` cleanup,
which would have forced a wasteful full re-run of a 3+ hour job.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import unittest

from syndiff_pipeline.template_creation.processing import ps1_process


class _FakeExecutorProcessesGoneAfterShutdown:
    """Mimics the real race: shutdown() nulls out _processes."""

    def __init__(self):
        self._processes = {}

    def shutdown(self, wait=False, cancel_futures=True):
        self._processes = None


class CleanupChildProcessesTests(unittest.TestCase):
    def setUp(self):
        self._orig_executor = ps1_process._active_executor
        self._orig_children = list(ps1_process._child_processes)

    def tearDown(self):
        ps1_process._active_executor = self._orig_executor
        ps1_process._child_processes[:] = self._orig_children

    def test_does_not_raise_when_processes_becomes_none_after_shutdown(self):
        ps1_process._active_executor = _FakeExecutorProcessesGoneAfterShutdown()
        ps1_process._child_processes[:] = []
        ps1_process._cleanup_child_processes()  # must not raise
        self.assertIsNone(ps1_process._active_executor)

    def test_still_terminates_processes_when_attribute_present(self):
        killed = []

        class FakeProc:
            def is_alive(self):
                return True

            def kill(self):
                killed.append(self)

        class FakeExecutorWithLiveProcesses:
            def __init__(self):
                self._processes = {1: FakeProc()}

            def shutdown(self, wait=False, cancel_futures=True):
                pass  # does NOT null out _processes this time

        ps1_process._active_executor = FakeExecutorWithLiveProcesses()
        ps1_process._child_processes[:] = []
        ps1_process._cleanup_child_processes()
        self.assertEqual(len(killed), 1)


if __name__ == "__main__":
    unittest.main()
