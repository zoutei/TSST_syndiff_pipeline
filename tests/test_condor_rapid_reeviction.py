"""Tests for the clean execute->evict eviction-loop detector.

Complements tests/test_condor_eviction_exclusion.py, which covers the
disconnect/"not found at execution machine" pattern. This covers a
different, equally real failure mode observed in production: a host's
startd cleanly re-evicting the same job over and over within seconds of
each (re)match -- plain "004 Job was evicted" events with no disconnect or
reconnect lines at all. Captured verbatim (host renamed) from a real
fix_missing_remap_s0020_c3_k3 / remap Condor log where a job was evicted
five times in a row, every ~60 seconds, always re-matched to the same host.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import unittest

from syndiff_pipeline.common.orchestration import condor


def _rapid_evict_cycle(cluster_id: int, host: str, hhmmss: str) -> str:
    """One clean execute -> evict cycle (real format, no disconnect lines)."""
    cid = f"{cluster_id:03d}.000.000"
    ts = f"2026-08-01 {hhmmss}"
    return "\n".join(
        [
            f"001 ({cid}) {ts} Job executing on host: "
            f"<10.128.72.78:9618?addrs=10.128.72.78-9618&alias={host}&noUDP&sock=startd_5049_c5c3>",
            f"\tSlotName: slot1_1@{host}",
            '\tCondorScratchDir = "/var/lib/condor/execute/dir_3576079"',
            "\tCpus = 64",
            "...",
            f"006 ({cid}) {ts} Image size of job updated: 64104",
            "\t63  -  MemoryUsage of job (MB)",
            "...",
            f"004 ({cid}) {ts} Job was evicted.",
            "\t(0) CPU times",
            "\t\tUsr 0 00:00:24, Sys 0 00:00:01  -  Run Remote Usage",
            "\t\tUsr 0 00:00:00, Sys 0 00:00:00  -  Run Local Usage",
            "\t0  -  Run Bytes Sent By Job",
            "\t0  -  Run Bytes Received By Job",
            "\tPartitionable Resources :    Usage  Request Allocated ",
            "\t   Memory (MB)          :       63   128000    500096 ",
            "...",
        ]
    )


def _success_cycle(cluster_id: int, host: str, hhmmss: str) -> str:
    """A job that starts and just keeps running (no eviction)."""
    cid = f"{cluster_id:03d}.000.000"
    ts = f"2026-08-01 {hhmmss}"
    return "\n".join(
        [
            f"001 ({cid}) {ts} Job executing on host: "
            f"<10.128.72.195:9618?addrs=10.128.72.195-9618&alias={host}&noUDP&sock=startd_2354_3a07>",
            f"\tSlotName: slot1_2@{host}",
            "\tCpus = 64",
            "...",
            f"006 ({cid}) {ts} Image size of job updated: 33720",
            "\t33  -  MemoryUsage of job (MB)",
            "...",
        ]
    )


class TallyRapidReevictionTests(unittest.TestCase):
    def test_counts_one_per_execute_evict_pair(self):
        log_text = "\n...\n".join(
            _rapid_evict_cycle(8, "plscience5.stsci.edu", f"13:{29 + i}:00")
            for i in range(5)
        )
        counts = condor.tally_rapid_reeviction_failures(log_text, cluster_id=8)
        self.assertEqual(counts, {"plscience5.stsci.edu": 5})

    def test_healthy_run_counts_nothing(self):
        log_text = _success_cycle(9, "plscience12.stsci.edu", "13:29:00")
        counts = condor.tally_rapid_reeviction_failures(log_text, cluster_id=9)
        self.assertEqual(counts, {})

    def test_scoped_to_requested_cluster_id(self):
        log_text = "\n...\n".join(
            [
                _rapid_evict_cycle(8, "plscience5.stsci.edu", "13:29:00"),
                _rapid_evict_cycle(8, "plscience5.stsci.edu", "13:30:00"),
                _rapid_evict_cycle(9, "plscience5.stsci.edu", "13:31:00"),
            ]
        )
        counts = condor.tally_rapid_reeviction_failures(log_text, cluster_id=8)
        self.assertEqual(counts, {"plscience5.stsci.edu": 2})

    def test_eviction_requeue_host_triggers_at_threshold(self):
        import tempfile

        log_text = "\n...\n".join(
            _rapid_evict_cycle(8, "plscience5.stsci.edu", f"13:{29 + i}:00")
            for i in range(2)
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "remap.condor.log"
            log_path.write_text(log_text, encoding="utf-8")
            state_path = Path(tmp) / "remap.condor.eviction_state"
            host = condor.eviction_requeue_host(
                log_path, cluster_id=8, eviction_state_path=state_path
            )
        self.assertEqual(host, "plscience5.stsci.edu")

    def test_combined_tallies_merges_both_detectors(self):
        # A rapid-reeviction cluster and a disconnect-cycle cluster, scoped
        # independently, should each surface via the combined tally.
        rapid = "\n...\n".join(
            _rapid_evict_cycle(8, "plscience5.stsci.edu", f"13:{29 + i}:00")
            for i in range(3)
        )
        counts = condor.combined_eviction_tallies(rapid, cluster_id=8)
        self.assertEqual(counts, {"plscience5.stsci.edu": 3})


if __name__ == "__main__":
    unittest.main()
