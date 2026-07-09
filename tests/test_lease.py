"""Tests for NFS ownership lease and stop-request helpers."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import lease, logs


class TestLeaseAcquire(unittest.TestCase):
    def test_first_acquire_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            owned = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.05)
            self.assertIsNotNone(owned)
            assert owned is not None
            self.assertEqual(owned.host, "host-a")
            self.assertEqual(owned.pid, 111)
            self.assertEqual(owned.generation, 1)
            self.assertTrue(owned.is_fresh())

    def test_second_acquire_loses_while_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.05)
            self.assertIsNotNone(first)
            second = lease.try_acquire_lease(tmp, host="host-b", pid=222, settle_s=0.05)
            self.assertIsNone(second)
            current = lease.read_lease(tmp)
            assert current is not None
            self.assertEqual(current.pid, 111)

    def test_stale_lease_allows_new_acquire(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.01)
            self.assertIsNotNone(first)
            assert first is not None
            stale = lease.Lease(
                host=first.host,
                pid=first.pid,
                generation=first.generation,
                started_at=first.started_at,
                renewed_at="2000-01-01T00:00:00+00:00",
            )
            lease.write_lease_atomic(tmp, stale)
            second = lease.try_acquire_lease(
                tmp, host="host-b", pid=222, settle_s=0.05, stale_after_s=1.0
            )
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.generation, first.generation + 1)
            self.assertEqual(second.host, "host-b")

    def test_same_host_dead_pid_reclaims_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.01)
            self.assertIsNotNone(first)
            with mock.patch(
                "syndiff_pipeline.common.orchestration.lease.daemon.local_hostname",
                return_value="host-a",
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.lease.daemon.is_process_alive",
                return_value=False,
            ):
                second = lease.try_acquire_lease(
                    tmp, host="host-a", pid=222, settle_s=0.05
                )
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.pid, 222)
            self.assertEqual(second.generation, 2)


class TestLeaseRenew(unittest.TestCase):
    def test_renew_updates_timestamp_not_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            owned = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.01)
            self.assertIsNotNone(owned)
            assert owned is not None
            time.sleep(0.02)
            renewed = lease.renew_lease(tmp, owned, host="host-a", pid=111)
            self.assertIsNotNone(renewed)
            assert renewed is not None
            self.assertEqual(renewed.generation, owned.generation)
            self.assertGreaterEqual(renewed.renewed_at, owned.renewed_at)

    def test_renew_fails_after_generation_stolen(self):
        with tempfile.TemporaryDirectory() as tmp:
            owned = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.01)
            self.assertIsNotNone(owned)
            assert owned is not None
            stolen = lease.Lease(
                host="host-b",
                pid=222,
                generation=owned.generation + 1,
                started_at=owned.started_at,
                renewed_at=owned.renewed_at,
            )
            lease.write_lease_atomic(tmp, stolen)
            renewed = lease.renew_lease(tmp, owned, host="host-a", pid=111)
            self.assertIsNone(renewed)


class TestStopRequest(unittest.TestCase):
    def test_stop_targets_matching_generation(self):
        owned = lease.Lease(
            host="host-a",
            pid=111,
            generation=7,
            started_at="2020-01-01T00:00:00+00:00",
            renewed_at="2020-01-01T00:00:00+00:00",
        )
        req = lease.StopRequest(
            requested_at="2020-01-01T00:01:00+00:00",
            requested_by_host="login02",
            target_generation=7,
        )
        self.assertTrue(lease.stop_targets_owner(req, owned))
        stale_req = lease.StopRequest(
            requested_at="2020-01-01T00:01:00+00:00",
            requested_by_host="login02",
            target_generation=6,
        )
        self.assertFalse(lease.stop_targets_owner(stale_req, owned))

    def test_write_and_clear_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = lease.write_stop_request(tmp, target_generation=3)
            self.assertEqual(req.target_generation, 3)
            self.assertTrue(logs.daemon_stop_path(tmp).is_file())
            self.assertTrue(lease.clear_stop_request(tmp, only_generation=3))
            self.assertTrue(logs.daemon_stop_path(tmp).is_file())
            self.assertIsNone(lease.read_stop_request(tmp))

    def test_stale_stop_cleared_on_acquire(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.01)
            assert first is not None
            lease.write_stop_request(tmp, target_generation=first.generation)
            stale = lease.Lease(
                host=first.host,
                pid=first.pid,
                generation=first.generation,
                started_at=first.started_at,
                renewed_at="2000-01-01T00:00:00+00:00",
            )
            lease.write_lease_atomic(tmp, stale)
            second = lease.try_acquire_lease(
                tmp, host="host-b", pid=222, settle_s=0.05, stale_after_s=1.0
            )
            self.assertIsNotNone(second)
            self.assertIsNone(lease.read_stop_request(tmp))
            self.assertTrue(logs.daemon_lease_path(tmp).is_file())

    def test_release_keeps_lease_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            owned = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.01)
            assert owned is not None
            self.assertTrue(lease.release_lease(tmp, host="host-a", pid=111, generation=owned.generation))
            self.assertTrue(logs.daemon_lease_path(tmp).is_file())
            self.assertIsNone(lease.read_lease(tmp))

    def test_wait_until_lease_released_on_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = lease.try_acquire_lease(tmp, host="host-a", pid=111, settle_s=0.01)
            assert first is not None
            stale = lease.Lease(
                host=first.host,
                pid=first.pid,
                generation=first.generation,
                started_at=first.started_at,
                renewed_at="2000-01-01T00:00:00+00:00",
            )
            lease.write_lease_atomic(tmp, stale)
            self.assertTrue(
                lease.wait_until_lease_released(
                    tmp,
                    target_generation=first.generation,
                    timeout_s=1.0,
                    stale_after_s=1.0,
                    poll_s=0.05,
                )
            )


if __name__ == "__main__":
    unittest.main()
