"""Regression tests for the bounded-wait fast-path verify wrapper.

Background
----------
``_run_verify_pass`` used to call ``_checkpoint_hit`` / ``check_manifests_only``
/ ``stage_absence_probe`` directly on the daemon's single main scheduling
thread. All three read from the shared NFS data root (provenance.db,
manifests, skycell CSVs); when NFS itself stalls (D-state, uninterruptible --
see the nfs-contention-plscience-cluster postmortem), the main thread blocked
indefinitely, freezing scheduling for every run on every host, not just the
one candidate that hit the stall.

``_fast_path_check_bounded`` now runs that work in a small dedicated thread
pool and only waits up to a bounded timeout for a same-tick result, falling
back to "not resolved yet" (``None``) rather than blocking the caller.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
import unittest.mock
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import scheduler
from syndiff_pipeline.common.orchestration.verify_worker import VerifyTaskKey


@dataclass(frozen=True)
class _FakeOutcome:
    complete: bool


class FastPathBoundedTests(unittest.TestCase):
    def setUp(self) -> None:
        scheduler.reset_fast_path_pool_for_tests()

    def tearDown(self) -> None:
        scheduler.reset_fast_path_pool_for_tests()

    def test_fast_result_returns_promptly(self):
        key = VerifyTaskKey("run1", "target1", "mapping")
        expected = scheduler._FastPathResult(outcome=_FakeOutcome(complete=True))
        with unittest.mock.patch.object(
            scheduler, "_run_fast_path_task", return_value=expected
        ):
            started = time.monotonic()
            result = scheduler._fast_path_check_bounded(
                key, None, "m", "s", None, None, False, timeout_s=2.0
            )
            elapsed = time.monotonic() - started
        self.assertIs(result, expected)
        self.assertLess(elapsed, 1.0)

    def test_stalled_check_times_out_without_blocking_past_timeout(self):
        key = VerifyTaskKey("run1", "target1", "ps1_process")
        release = threading.Event()

        def _blocking_task(*_args, **_kwargs):
            release.wait()  # simulates an NFS D-state stall
            return scheduler._FastPathResult(needs_full_verify=True)

        with unittest.mock.patch.object(
            scheduler, "_run_fast_path_task", side_effect=_blocking_task
        ):
            started = time.monotonic()
            result = scheduler._fast_path_check_bounded(
                key, None, "m", "s", None, None, False, timeout_s=0.3
            )
            elapsed = time.monotonic() - started
            self.assertIsNone(result)
            self.assertLess(elapsed, 1.0, "bounded wait should not exceed timeout_s by much")

            # Re-encountering the same key (e.g. next tick) must not re-block:
            # it's already in flight, so this should return near-instantly.
            started2 = time.monotonic()
            result2 = scheduler._fast_path_check_bounded(
                key, None, "m", "s", None, None, False, timeout_s=5.0
            )
            elapsed2 = time.monotonic() - started2
            self.assertIsNone(result2)
            self.assertLess(elapsed2, 0.5, "already-in-flight key must be a near-zero poll")

            release.set()
            # Give the worker thread a moment to finish and be drained.
            deadline = time.monotonic() + 2.0
            final = None
            while time.monotonic() < deadline:
                final = scheduler._fast_path_check_bounded(
                    key, None, "m", "s", None, None, False, timeout_s=0.5
                )
                if final is not None:
                    break
                time.sleep(0.02)
        self.assertIsNotNone(final)
        self.assertTrue(final.needs_full_verify)
        # The dedup entry must be cleared once resolved.
        self.assertNotIn(key, scheduler._fast_path_in_flight)

    def test_distinct_keys_do_not_interfere(self):
        key_a = VerifyTaskKey("run1", "targetA", "mapping")
        key_b = VerifyTaskKey("run1", "targetB", "mapping")
        release = threading.Event()

        def _task(key, *_args, **_kwargs):
            if key == key_a:
                release.wait()
            return scheduler._FastPathResult(outcome=_FakeOutcome(complete=True))

        with unittest.mock.patch.object(scheduler, "_run_fast_path_task", side_effect=_task):
            result_a = scheduler._fast_path_check_bounded(
                key_a, None, "m", "s", None, None, False, timeout_s=0.2
            )
            self.assertIsNone(result_a)
            result_b = scheduler._fast_path_check_bounded(
                key_b, None, "m", "s", None, None, False, timeout_s=2.0
            )
            self.assertIsNotNone(result_b)
            self.assertTrue(result_b.outcome.complete)
            release.set()


if __name__ == "__main__":
    unittest.main()
