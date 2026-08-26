"""Extensive tests for Condor immediate-evict detection and bad-machine exclusion.

The log fixtures in this file are built to byte-for-byte match the real
HTCondor user-log event format observed in production (a job matching a
vanished/broken slot on ``plscience10.stsci.edu``, captured from
``runs/star_lc_verify/per_target/s0020_c3_k2_s20_astrometry/star.condor.log``).
That real log was used to validate the parser during development: it
contains three retried clusters (003, 004, 005) all evicted in a loop on
the same bad host, plus a fourth cluster (006) that matched a good host and
ran normally. Any change to the parsing regexes should be re-validated
against a real captured log, not just synthetic fixtures.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import condor, logs
from syndiff_pipeline.common.orchestration.run_context import resolve_run_context
from syndiff_pipeline.common.orchestration.scheduler import reconcile_running_stages
from syndiff_pipeline.common.orchestration.state import (
    PipelineState,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_RUNNING,
)
from syndiff_pipeline.common.orchestration.targets import Target
from tests.test_condor_lifecycle import _minimal_condor_run


def _disconnect_cycle(cluster_id: int, host: str, hhmmss: str) -> str:
    """One disconnect -> reconnect-failed -> evicted cycle, in the exact
    format HTCondor writes to the user log for a job matched to a slot that
    has vanished (verified against a real production log)."""
    cid = f"{cluster_id:03d}.000.000"
    ts = f"2026-07-09 {hhmmss}"
    return "\n".join(
        [
            f"022 ({cid}) {ts} Job disconnected, attempting to reconnect",
            "    Socket between submit and execute hosts closed unexpectedly",
            f"    Trying to reconnect to slot1_1@{host} "
            f"<10.128.72.93:9618?addrs=10.128.72.93-9618&alias={host}&noUDP&sock=startd_2951_b811>",
            "...",
            f"024 ({cid}) {ts} Job reconnection failed",
            "    Job not found at execution machine",
            f"    Can not reconnect to slot1_1@{host}, rescheduling job",
            "...",
            f"004 ({cid}) {ts} Job was evicted.",
            "\t(0) CPU times",
            "\t\tUsr 0 00:00:00, Sys 0 00:00:00  -  Run Remote Usage",
            "\t\tUsr 0 00:00:00, Sys 0 00:00:00  -  Run Local Usage",
            "\t0  -  Run Bytes Sent By Job",
            "\t0  -  Run Bytes Received By Job",
            "\tJob not found at execution machine",
            "\tPartitionable Resources :    Usage  Request Allocated ",
            "\t   Cpus                 :                 8         8 ",
            "...",
        ]
    )


def _eviction_log(
    cluster_id: int,
    host: str,
    *,
    failures: int,
    start_hour: int = 16,
    start_minute: int = 0,
) -> str:
    blocks = []
    for i in range(failures):
        minute = (start_minute + i) % 60
        hour = start_hour + (start_minute + i) // 60
        blocks.append(_disconnect_cycle(cluster_id, host, f"{hour:02d}:{minute:02d}:00"))
    return "\n...\n".join(blocks)


def _submitted_line(cluster_id: int, hhmmss: str) -> str:
    cid = f"{cluster_id:03d}.000.000"
    return (
        f"000 ({cid}) 2026-07-09 {hhmmss} Job submitted from host: "
        "<10.128.72.78:9618?addrs=10.128.72.78-9618&alias=plscience5.stsci.edu&noUDP&sock=schedd_2362_b6c9>"
    )


def _success_block(cluster_id: int, host: str, hhmmss: str) -> str:
    """A job that matches a healthy host and actually starts running."""
    cid = f"{cluster_id:03d}.000.000"
    ts = f"2026-07-09 {hhmmss}"
    return "\n".join(
        [
            f"001 ({cid}) {ts} Job executing on host: "
            f"<10.128.72.195:9618?addrs=10.128.72.195-9618&alias={host}&noUDP&sock=startd_2354_3a07>",
            f"\tSlotName: slot1_2@{host}",
            '\tCondorScratchDir = "/var/lib/condor/execute/dir_108038"',
            "\tCpus = 8",
            "...",
            f"006 ({cid}) {ts} Image size of job updated: 33720",
            "\t33  -  MemoryUsage of job (MB)",
            "...",
        ]
    )


def _aborted_line(cluster_id: int, hhmmss: str) -> str:
    cid = f"{cluster_id:03d}.000.000"
    return "\n".join(
        [
            f"009 ({cid}) 2026-07-09 {hhmmss} Job was aborted.",
            "\tvia condor_rm (by user kshukawa)",
            "...",
        ]
    )


def _set_stage_fields(
    state: PipelineState,
    run_id: str,
    label: str,
    stage: str,
    **fields,
) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [run_id, label, stage]
    with state._conn() as conn:
        conn.execute(
            f"UPDATE stage_runs SET {sets} WHERE run_id = ? AND target_label = ? AND stage = ?",
            values,
        )


def _claim_condor_stage(
    tmp_path: Path,
    *,
    cluster_id: int,
    stage: str = "star",
    target_name: str = "s20_astrometry",
    condor_log: str,
    stage_log_age_s: float = 3600.0,
    attempts: int | None = None,
) -> tuple[PipelineState, object, str, dict[str, Path]]:
    target = Target(
        sector=20,
        camera=3,
        ccd=2,
        target_ra=210.0,
        target_dec=81.0,
        target_name=target_name,
    )
    state, run_dir = _minimal_condor_run(tmp_path, target)
    label = target.label()
    runs_root = tmp_path / "runs"
    log_path = logs.target_log_path(str(runs_root), "run_a", label, stage)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("starting\n", encoding="utf-8")
    old = time.time() - stage_log_age_s
    os.utime(log_path, (old, old))
    artifacts = condor.condor_artifact_paths(str(runs_root), "run_a", label, stage)
    artifacts["log"].write_text(condor_log, encoding="utf-8")
    state.update_stage_status("run_a", label, stage, STATUS_READY)
    state.try_atomic_claim(
        "run_a",
        label,
        stage,
        launch_token=state.new_launch_token(),
        executor="condor",
        native_id=cluster_id,
        log_path=str(log_path),
        submit_epoch=time.time(),
    )
    if attempts is not None:
        _set_stage_fields(state, "run_a", label, stage, attempts=attempts)
    ctx = resolve_run_context(run_dir=run_dir)
    return state, ctx, label, artifacts


class TestCondorHostNormalization(unittest.TestCase):
    def test_slot_prefix_stripped(self):
        self.assertEqual(
            condor.normalize_condor_host("slot1_1@plscience10.stsci.edu"),
            "plscience10.stsci.edu",
        )

    def test_partitionable_slot_prefix_stripped(self):
        # Real slots are often "slot1_1@host" (dynamic slot under a
        # partitionable parent); make sure the whole prefix goes, not just
        # up to the first underscore.
        self.assertEqual(
            condor.normalize_condor_host("slot1_1_1@plscience10.stsci.edu"),
            "plscience10.stsci.edu",
        )

    def test_bare_hostname_passthrough(self):
        self.assertEqual(
            condor.normalize_condor_host("plscience15.stsci.edu"),
            "plscience15.stsci.edu",
        )

    def test_empty_string(self):
        self.assertEqual(condor.normalize_condor_host(""), "")

    def test_none_like_input(self):
        self.assertEqual(condor.normalize_condor_host(None), "")  # type: ignore[arg-type]

    def test_whitespace_trimmed(self):
        self.assertEqual(
            condor.normalize_condor_host("  plscience10.stsci.edu  "),
            "plscience10.stsci.edu",
        )

    def test_multiple_at_signs_uses_last_segment(self):
        self.assertEqual(
            condor.normalize_condor_host("user@slot1_1@plscience10.stsci.edu"),
            "plscience10.stsci.edu",
        )


class TestCondorEvictionLogParsingRealFormat(unittest.TestCase):
    """Parser tests built on the exact real HTCondor event sequence."""

    def test_single_cycle_counts_one(self):
        log = _disconnect_cycle(5, "plscience10.stsci.edu", "16:59:20")
        self.assertEqual(
            condor.tally_execution_eviction_failures(log),
            {"plscience10.stsci.edu": 1},
        )

    def test_two_cycles_same_host_counts_two(self):
        log = _eviction_log(5, "plscience10.stsci.edu", failures=2)
        self.assertEqual(
            condor.tally_execution_eviction_failures(log),
            {"plscience10.stsci.edu": 2},
        )

    def test_not_found_appears_twice_per_cycle_but_counted_once(self):
        # Sanity check on the fixture itself: "not found at execution
        # machine" appears twice within a single cycle (once under the 024
        # event, once under the 004 event) but must only be tallied once.
        log = _disconnect_cycle(5, "plscience10.stsci.edu", "16:59:20")
        self.assertEqual(log.count("not found at execution machine"), 2)
        self.assertEqual(
            condor.tally_execution_eviction_failures(log)["plscience10.stsci.edu"],
            1,
        )

    def test_many_repeated_cycles_accumulate(self):
        log = _eviction_log(5, "plscience10.stsci.edu", failures=17)
        self.assertEqual(
            condor.tally_execution_eviction_failures(log)["plscience10.stsci.edu"],
            17,
        )

    def test_successful_run_has_no_failures(self):
        log = "\n".join(
            [
                _submitted_line(6, "17:16:57"),
                "...",
                _success_block(6, "plscience12.stsci.edu", "17:17:05"),
            ]
        )
        self.assertEqual(condor.tally_execution_eviction_failures(log), {})

    def test_full_production_scenario_three_bad_clusters_one_good(self):
        """Reproduces the real production log structure: three retried
        clusters evicted in a loop on plscience10, then a fourth cluster
        that matched a healthy host and ran fine."""
        log = "\n...\n".join(
            [
                _submitted_line(3, "15:52:58"),
                _eviction_log(3, "plscience10.stsci.edu", failures=17, start_hour=15, start_minute=54),
                _aborted_line(3, "16:30:47"),
                _submitted_line(4, "16:32:51"),
                _eviction_log(4, "plscience10.stsci.edu", failures=11, start_hour=16, start_minute=48),
                _aborted_line(4, "16:58:45"),
                _submitted_line(5, "16:59:15"),
                _eviction_log(5, "plscience10.stsci.edu", failures=11, start_hour=16, start_minute=59),
                _aborted_line(5, "17:15:40"),
                _submitted_line(6, "17:16:57"),
                _success_block(6, "plscience12.stsci.edu", "17:17:05"),
            ]
        )
        # Unfiltered, cross-cluster tally sums everything in the file.
        self.assertEqual(
            condor.tally_execution_eviction_failures(log),
            {"plscience10.stsci.edu": 39},
        )
        # But scoped per cluster (the production-critical behavior), each
        # cluster's own count is isolated from the others.
        self.assertEqual(
            condor.tally_execution_eviction_failures(log, cluster_id=3)[
                "plscience10.stsci.edu"
            ],
            17,
        )
        self.assertEqual(
            condor.tally_execution_eviction_failures(log, cluster_id=4)[
                "plscience10.stsci.edu"
            ],
            11,
        )
        self.assertEqual(
            condor.tally_execution_eviction_failures(log, cluster_id=5)[
                "plscience10.stsci.edu"
            ],
            11,
        )
        self.assertEqual(condor.tally_execution_eviction_failures(log, cluster_id=6), {})

    def test_multiple_hosts_in_same_cluster_tallied_separately(self):
        log = "\n...\n".join(
            [
                _disconnect_cycle(5, "plscience10.stsci.edu", "16:00:00"),
                _disconnect_cycle(5, "plscience11.stsci.edu", "16:01:00"),
                _disconnect_cycle(5, "plscience10.stsci.edu", "16:02:00"),
            ]
        )
        self.assertEqual(
            condor.tally_execution_eviction_failures(log, cluster_id=5),
            {"plscience10.stsci.edu": 2, "plscience11.stsci.edu": 1},
        )

    def test_disconnect_without_reconnect_failure_not_counted(self):
        # "022 disconnected" followed by a successful reconnect (023) should
        # not be treated as a failure.
        log = "\n".join(
            [
                "022 (005.000.000) 2026-07-09 16:00:00 Job disconnected, attempting to reconnect",
                "    Trying to reconnect to slot1_1@plscience10.stsci.edu <...>",
                "...",
                "023 (005.000.000) 2026-07-09 16:00:05 Job reconnected to plscience10.stsci.edu",
            ]
        )
        self.assertEqual(condor.tally_execution_eviction_failures(log), {})

    def test_reconnection_failed_for_other_reason_not_counted(self):
        # A reconnection failure whose text does *not* mention "not found at
        # execution machine" (e.g. lease expiry from a slow network) should
        # not be attributed to a bad machine; it may be transient.
        log = "\n".join(
            [
                "022 (005.000.000) 2026-07-09 16:00:00 Job disconnected, attempting to reconnect",
                "    Trying to reconnect to slot1_1@plscience10.stsci.edu <...>",
                "...",
                "024 (005.000.000) 2026-07-09 16:20:00 Job reconnection failed",
                "    Job disconnected too long: JobLeaseDuration (1200 seconds) expired",
            ]
        )
        self.assertEqual(condor.tally_execution_eviction_failures(log), {})

    def test_preemption_eviction_without_disconnect_not_counted(self):
        # A job that ran fine and was later preempted/evicted (checkpoint,
        # priority) without ever going through the disconnect/reconnect
        # cycle must not be flagged as a bad-machine loop.
        log = "\n".join(
            [
                _success_block(6, "plscience12.stsci.edu", "17:17:05"),
                "004 (006.000.000) 2026-07-09 18:00:00 Job was evicted.",
                "\tJob was not found at execution machine",  # unlikely but even so:
            ]
        )
        self.assertEqual(condor.tally_execution_eviction_failures(log), {})

    def test_case_insensitive_not_found(self):
        log = "\n".join(
            [
                "022 (005.000.000) 2026-07-09 16:00:00 Job disconnected, attempting to reconnect",
                "    Trying to reconnect to slot1_1@plscience10.stsci.edu <...>",
                "...",
                "024 (005.000.000) 2026-07-09 16:00:01 Job reconnection failed",
                "    JOB NOT FOUND AT EXECUTION MACHINE",
            ]
        )
        self.assertEqual(
            condor.tally_execution_eviction_failures(log),
            {"plscience10.stsci.edu": 1},
        )

    def test_empty_log_returns_empty_dict(self):
        self.assertEqual(condor.tally_execution_eviction_failures(""), {})

    def test_garbled_binary_like_content_does_not_raise(self):
        log = "\x00\x01garbage\ufffd\nnot found at execution machine\n"
        self.assertEqual(condor.tally_execution_eviction_failures(log), {})

    def test_windows_line_endings_still_parse(self):
        log = _disconnect_cycle(5, "plscience10.stsci.edu", "16:00:00").replace("\n", "\r\n")
        self.assertEqual(
            condor.tally_execution_eviction_failures(log),
            {"plscience10.stsci.edu": 1},
        )

    def test_cluster_filter_with_no_matching_events_returns_empty(self):
        log = _disconnect_cycle(5, "plscience10.stsci.edu", "16:00:00")
        self.assertEqual(condor.tally_execution_eviction_failures(log, cluster_id=999), {})

    def test_missing_header_lines_default_to_unfiltered_when_no_cluster_given(self):
        # Continuation-only text (no "NNN (cid.p.s)" headers at all) should
        # still be parsed when no cluster filter is requested.
        log = "\n".join(
            [
                "Job disconnected, attempting to reconnect",
                "Trying to reconnect to slot1_1@plscience10.stsci.edu <...>",
                "Job reconnection failed",
                "Job not found at execution machine",
            ]
        )
        self.assertEqual(
            condor.tally_execution_eviction_failures(log),
            {"plscience10.stsci.edu": 1},
        )


class TestCondorEvictionRequeueDecision(unittest.TestCase):
    def test_missing_log_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.log"
            state_path = Path(tmp) / "state.json"
            self.assertIsNone(
                condor.eviction_requeue_host(
                    path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                )
            )

    def test_empty_log_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            log_path.write_text("", encoding="utf-8")
            state_path = Path(tmp) / "state.json"
            self.assertIsNone(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                )
            )

    def test_one_failure_below_default_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=1), encoding="utf-8"
            )
            self.assertIsNone(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                )
            )

    def test_two_failures_triggers_host_at_default_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=2), encoding="utf-8"
            )
            self.assertEqual(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                ),
                "plscience10.stsci.edu",
            )

    def test_custom_threshold_of_three_requires_three_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=2), encoding="utf-8"
            )
            self.assertIsNone(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                    threshold=3,
                )
            )
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=3), encoding="utf-8"
            )
            self.assertEqual(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                    threshold=3,
                ),
                "plscience10.stsci.edu",
            )

    def test_recorded_action_suppresses_repeat_for_same_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=2), encoding="utf-8"
            )
            condor.record_eviction_requeue(
                state_path,
                cluster_id=5,
                host="plscience10.stsci.edu",
                failure_count=2,
            )
            self.assertIsNone(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                )
            )

    def test_new_failures_after_action_trigger_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            condor.record_eviction_requeue(
                state_path,
                cluster_id=5,
                host="plscience10.stsci.edu",
                failure_count=2,
            )
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=4), encoding="utf-8"
            )
            self.assertEqual(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                ),
                "plscience10.stsci.edu",
            )

    def test_new_cluster_id_is_not_punished_by_old_clusters_history(self):
        """Regression test for the log-file-reuse bug: the same log file is
        appended across every retry/cluster. A brand-new cluster that hasn't
        failed yet must not inherit an older cluster's failure count."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            old_cluster_log = "\n...\n".join(
                [
                    _submitted_line(3, "15:52:58"),
                    _eviction_log(
                        3, "plscience10.stsci.edu", failures=17, start_hour=15, start_minute=54
                    ),
                    _aborted_line(3, "16:30:47"),
                ]
            )
            new_cluster_log = "\n...\n".join(
                [_submitted_line(4, "16:32:51"), _success_block(4, "plscience13.stsci.edu", "16:33:00")]
            )
            log_path.write_text(old_cluster_log + "\n...\n" + new_cluster_log, encoding="utf-8")
            self.assertIsNone(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=4,
                    eviction_state_path=state_path,
                )
            )

    def test_different_clusters_tracked_independently_in_acted_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            log_path.write_text(
                _eviction_log(6, "plscience10.stsci.edu", failures=2), encoding="utf-8"
            )
            condor.record_eviction_requeue(
                state_path,
                cluster_id=5,
                host="plscience10.stsci.edu",
                failure_count=2,
            )
            self.assertEqual(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=6,
                    eviction_state_path=state_path,
                ),
                "plscience10.stsci.edu",
            )

    def test_corrupt_eviction_state_still_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=2), encoding="utf-8"
            )
            state_path.write_text("{not json", encoding="utf-8")
            self.assertEqual(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                ),
                "plscience10.stsci.edu",
            )

    def test_eviction_state_list_instead_of_dict_recovers_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state_path = Path(tmp) / "state.json"
            log_path.write_text(
                _eviction_log(5, "plscience10.stsci.edu", failures=2), encoding="utf-8"
            )
            state_path.write_text(json.dumps(["unexpected", "list"]), encoding="utf-8")
            self.assertEqual(
                condor.eviction_requeue_host(
                    log_path,
                    cluster_id=5,
                    eviction_state_path=state_path,
                ),
                "plscience10.stsci.edu",
            )


class TestCondorBadMachinesPersistence(unittest.TestCase):
    def test_roundtrip_read_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            condor.write_bad_machines(path, {"plscience10.stsci.edu", "plscience11.stsci.edu"})
            self.assertEqual(
                condor.read_bad_machines(path),
                {"plscience10.stsci.edu", "plscience11.stsci.edu"},
            )

    def test_add_bad_machine_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            self.assertTrue(condor.add_bad_machine(path, "slot1@plscience10.stsci.edu"))
            self.assertFalse(condor.add_bad_machine(path, "plscience10.stsci.edu"))
            self.assertEqual(condor.read_bad_machines(path), {"plscience10.stsci.edu"})

    def test_add_bad_machine_empty_host_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            self.assertFalse(condor.add_bad_machine(path, ""))
            self.assertFalse(path.is_file())

    def test_corrupt_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(condor.read_bad_machines(path), set())

    def test_hosts_not_a_list_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"hosts": "not-a-list"}), encoding="utf-8")
            self.assertEqual(condor.read_bad_machines(path), set())

    def test_bare_list_payload_also_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(["plscience10.stsci.edu"]), encoding="utf-8")
            self.assertEqual(condor.read_bad_machines(path), {"plscience10.stsci.edu"})

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(condor.read_bad_machines(Path(tmp) / "nope.json"), set())

    def test_artifact_paths_include_bad_machines_and_eviction_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = condor.condor_artifact_paths(str(Path(tmp) / "runs"), "run_a", "target", "star")
            self.assertIn("bad_machines", paths)
            self.assertIn("eviction_state", paths)
            self.assertTrue(str(paths["bad_machines"]).endswith("star.condor.bad_machines"))
            self.assertTrue(str(paths["eviction_state"]).endswith("star.condor.eviction_state"))

    def test_bad_machines_persist_across_process_restart_simulation(self):
        # Module-level in-memory caches (_submission_times/_held_times) are
        # lost on daemon restart; bad_machines must be file-backed only.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            condor.add_bad_machine(path, "plscience10.stsci.edu")
            # Re-read via a brand-new call with no shared in-memory state.
            self.assertEqual(condor.read_bad_machines(path), {"plscience10.stsci.edu"})


class TestCondorRequirementsExclusion(unittest.TestCase):
    def test_no_exclusions_returns_original(self):
        self.assertEqual(
            condor.merge_requirements_with_exclusions("LoadAvg < 10", set()),
            "LoadAvg < 10",
        )

    def test_none_requirements_becomes_true_with_exclusion(self):
        merged = condor.merge_requirements_with_exclusions(None, {"plscience10.stsci.edu"})
        self.assertEqual(merged, '(True) && Machine != "plscience10.stsci.edu"')

    def test_empty_string_requirements_becomes_true_with_exclusion(self):
        merged = condor.merge_requirements_with_exclusions("", {"plscience10.stsci.edu"})
        self.assertEqual(merged, '(True) && Machine != "plscience10.stsci.edu"')

    def test_multiple_hosts_sorted_deterministically(self):
        merged = condor.merge_requirements_with_exclusions(
            "Memory >= 100000",
            {"plscience11.stsci.edu", "plscience10.stsci.edu"},
        )
        self.assertEqual(
            merged,
            '(Memory >= 100000) && Machine != "plscience10.stsci.edu" && '
            'Machine != "plscience11.stsci.edu"',
        )
        # Order of the input set must not affect output (frozenset iteration
        # order is unspecified).
        merged_reordered = condor.merge_requirements_with_exclusions(
            "Memory >= 100000",
            {"plscience10.stsci.edu", "plscience11.stsci.edu"},
        )
        self.assertEqual(merged, merged_reordered)

    def test_apply_exclusions_returns_new_frozen_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = condor.condor_artifact_paths(str(Path(tmp) / "runs"), "r", "t", "star")
            condor.write_bad_machines(artifacts["bad_machines"], {"plscience10.stsci.edu"})
            base = condor.CondorResourceRequest(requirements="LoadAvg < 10")
            merged = condor.apply_bad_machine_exclusions(base, artifacts)
            self.assertIsNot(merged, base)
            self.assertIn("plscience10.stsci.edu", merged.requirements or "")
            self.assertEqual(base.requirements, "LoadAvg < 10")

    def test_apply_exclusions_preserves_other_resource_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = condor.condor_artifact_paths(str(Path(tmp) / "runs"), "r", "t", "star")
            condor.write_bad_machines(artifacts["bad_machines"], {"plscience10.stsci.edu"})
            base = condor.CondorResourceRequest(
                request_cpus=16,
                request_memory_mb=64000,
                requirements="LoadAvg < 10",
                rank="-LoadAvg",
            )
            merged = condor.apply_bad_machine_exclusions(base, artifacts)
            self.assertEqual(merged.request_cpus, 16)
            self.assertEqual(merged.request_memory_mb, 64000)
            self.assertEqual(merged.rank, "-LoadAvg")

    def test_no_bad_machines_file_returns_same_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = condor.condor_artifact_paths(str(Path(tmp) / "runs"), "r", "t", "star")
            base = condor.CondorResourceRequest(requirements="LoadAvg < 10")
            merged = condor.apply_bad_machine_exclusions(base, artifacts)
            self.assertIs(merged, base)


class TestCondorSubmitExclusions(unittest.TestCase):
    def _empty_host_stats_env(self, tmp: str):
        stats_dir = Path(tmp) / "empty_host_stats"
        stats_dir.mkdir()
        return unittest.mock.patch.dict(os.environ, {"HOST_STATS_DIR": str(stats_dir)})

    def test_submit_job_merges_bad_machines_into_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            artifacts = condor.condor_artifact_paths(runs_root, "run_a", "target", "star")
            condor.write_bad_machines(artifacts["bad_machines"], {"plscience10.stsci.edu"})
            proc = unittest.mock.Mock(
                stdout="submitted to cluster 42",
                stderr="",
                returncode=0,
            )
            with self._empty_host_stats_env(tmp), unittest.mock.patch.object(
                condor, "_run_condor", return_value=proc
            ):
                cluster_id, _epoch = condor.submit_job(
                    ["echo", "hi"],
                    runs_root,
                    "run_a",
                    "target",
                    "star",
                    resources=condor.CondorResourceRequest(requirements="LoadAvg < 10"),
                )
            self.assertEqual(cluster_id, 42)
            submit_text = artifacts["submit"].read_text(encoding="utf-8")
            self.assertIn('Machine != "plscience10.stsci.edu"', submit_text)
            self.assertIn("Memory >= 500000", submit_text)

    def test_submit_without_bad_machines_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            artifacts = condor.condor_artifact_paths(runs_root, "run_a", "target", "star")
            proc = unittest.mock.Mock(
                stdout="submitted to cluster 43",
                stderr="",
                returncode=0,
            )
            with self._empty_host_stats_env(tmp), unittest.mock.patch.object(
                condor, "_run_condor", return_value=proc
            ):
                condor.submit_job(
                    ["echo", "hi"],
                    runs_root,
                    "run_a",
                    "target",
                    "star",
                    resources=condor.CondorResourceRequest(requirements="LoadAvg < 10"),
                )
            submit_text = artifacts["submit"].read_text(encoding="utf-8")
            self.assertEqual(submit_text.count("Machine !="), 0)
            self.assertIn("requirements = Memory >= 500000", submit_text)
            self.assertIn("rank = -LoadAvg", submit_text)

    def test_submit_with_default_resources_and_no_bad_machines(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            artifacts = condor.condor_artifact_paths(runs_root, "run_a", "target", "star")
            proc = unittest.mock.Mock(
                stdout="submitted to cluster 44",
                stderr="",
                returncode=0,
            )
            with self._empty_host_stats_env(tmp), unittest.mock.patch.object(
                condor, "_run_condor", return_value=proc
            ):
                cluster_id, _epoch = condor.submit_job(
                    ["echo", "hi"], runs_root, "run_a", "target", "star"
                )
            self.assertEqual(cluster_id, 44)


class TestCondorEvictionReconcile(unittest.TestCase):
    """Full reconcile-loop tests against the real HTCondor log format,
    exercising the exact production scenario end-to-end."""

    def test_idle_job_with_one_failure_stays_running(self):
        cluster_id = 900_001
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, _artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=1),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster") as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(counts["requeued"], 0)
            rm.assert_not_called()
            row = state.get_stage_run("run_a", label, "star")
            self.assertEqual(row.status, STATUS_RUNNING)

    def test_running_job_with_two_failures_requeues(self):
        cluster_id = 900_002
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=2),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_RUNNING, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True) as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["requeued"], 1)
            self.assertEqual(counts["still_running"], 0)
            rm.assert_called_once()
            row = state.get_stage_run("run_a", label, "star")
            self.assertEqual(row.status, STATUS_READY)
            self.assertIn("plscience10.stsci.edu", condor.read_bad_machines(artifacts["bad_machines"]))

    def test_active_stage_log_defers_eviction_requeue(self):
        cluster_id = 900_003
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=5),
                stage_log_age_s=30.0,
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster") as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(counts["requeued"], 0)
            rm.assert_not_called()
            self.assertFalse(artifacts["bad_machines"].is_file())

    def test_second_reconcile_does_not_double_requeue(self):
        cluster_id = 900_004
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=2),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True):
                first = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(first["requeued"], 1)
            # Simulate relaunch on same cluster id without new log lines
            # (e.g. condor_rm hasn't fully taken effect yet on a re-poll).
            state.try_atomic_claim(
                "run_a",
                label,
                "star",
                launch_token=state.new_launch_token(),
                executor="condor",
                native_id=cluster_id,
                log_path=str(
                    logs.target_log_path(str(Path(tmp) / "runs"), "run_a", label, "star")
                ),
                submit_epoch=time.time(),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True) as rm:
                second = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(second["requeued"], 0)
            self.assertEqual(second["still_running"], 1)
            rm.assert_not_called()
            self.assertTrue(artifacts["eviction_state"].is_file())

    def test_max_attempts_marks_failed_instead_of_requeue(self):
        """Eviction-loop requeues are budgeted against max_eviction_stage_attempts,
        not the tighter generic max_stage_attempts -- a bad/flaky host getting
        excluded and requeued is expected recovery, not stage failure, and
        sharing the small default budget with genuine launch failures can
        exhaust it before ever reaching a good host in a small pool."""
        cluster_id = 900_005
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=2),
                attempts=3,
            )
            ctx.cfg.max_stage_attempts = 1000
            ctx.cfg.max_eviction_stage_attempts = 3
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True):
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(counts["requeued"], 0)
            row = state.get_stage_run("run_a", label, "star")
            self.assertEqual(row.status, STATUS_FAILED)
            self.assertIn("plscience10.stsci.edu", condor.read_bad_machines(artifacts["bad_machines"]))

    def test_eviction_requeue_survives_past_tight_generic_max_attempts(self):
        """A small pool exploring bad hosts must not exhaust the generic
        max_stage_attempts budget: eviction-loop requeues use the larger,
        dedicated max_eviction_stage_attempts allowance instead."""
        cluster_id = 900_005
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience12.stsci.edu", failures=2),
                attempts=3,
            )
            ctx.cfg.max_stage_attempts = 3
            ctx.cfg.max_eviction_stage_attempts = 20
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True):
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["requeued"], 1)
            self.assertEqual(counts["failed"], 0)
            row = state.get_stage_run("run_a", label, "star")
            self.assertEqual(row.status, STATUS_READY)
            self.assertIn("plscience12.stsci.edu", condor.read_bad_machines(artifacts["bad_machines"]))

    def test_held_job_not_treated_as_eviction_loop(self):
        cluster_id = 900_006
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, _artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=2),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_HELD, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster") as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(counts["requeued"], 0)
            rm.assert_not_called()

    def test_completed_job_not_requeued_for_old_evictions(self):
        cluster_id = 900_007
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, _artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=2),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_COMPLETED, 0)},
            ), unittest.mock.patch.object(condor, "remove_cluster") as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["completed"], 1)
            self.assertEqual(counts["requeued"], 0)
            rm.assert_not_called()

    def test_removed_job_not_requeued_for_old_evictions(self):
        cluster_id = 900_010
        with tempfile.TemporaryDirectory() as tmp:
            state, ctx, label, _artifacts = _claim_condor_stage(
                Path(tmp),
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=2),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_REMOVED, 0)},
            ), unittest.mock.patch.object(condor, "remove_cluster") as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            # STATUS_CANCELED path; the key point is no eviction-exclusion
            # requeue is attempted for a job that's already gone.
            self.assertEqual(counts["requeued"], 0)
            rm.assert_not_called()

    def test_new_cluster_after_requeue_does_not_reinherit_old_failures(self):
        """End-to-end regression test for the log-reuse cluster-scoping bug:
        after cluster A is excluded/requeued, cluster B (a fresh submission
        reusing the same log file) must not be immediately requeued again
        just because cluster A's old failures are still in the file."""
        cluster_a = 900_011
        cluster_b = 900_012
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, label, artifacts = _claim_condor_stage(
                tmp_path,
                cluster_id=cluster_a,
                condor_log=_eviction_log(cluster_a, "plscience10.stsci.edu", failures=2),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_a: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True):
                first = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(first["requeued"], 1)

            # New cluster submitted (its own submit_job call would already
            # exclude plscience10.stsci.edu via apply_bad_machine_exclusions,
            # but simulate the log-append behavior regardless: append a
            # *successful* match for cluster B onto the SAME log file.
            with artifacts["log"].open("a", encoding="utf-8") as fh:
                fh.write("\n...\n")
                fh.write(_submitted_line(cluster_b, "18:00:00"))
                fh.write("\n...\n")
                fh.write(_success_block(cluster_b, "plscience13.stsci.edu", "18:00:05"))
            state.try_atomic_claim(
                "run_a",
                label,
                "star",
                launch_token=state.new_launch_token(),
                executor="condor",
                native_id=cluster_b,
                log_path=str(
                    logs.target_log_path(str(tmp_path / "runs"), "run_a", label, "star")
                ),
                submit_epoch=time.time(),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_b: (condor._JOB_RUNNING, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True) as rm:
                second = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(second["requeued"], 0)
            self.assertEqual(second["still_running"], 1)
            rm.assert_not_called()
            row = state.get_stage_run("run_a", label, "star")
            self.assertEqual(row.status, STATUS_RUNNING)

    def test_accumulates_multiple_bad_hosts_over_retries(self):
        cluster_id = 900_008
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, label, artifacts = _claim_condor_stage(
                tmp_path,
                cluster_id=cluster_id,
                condor_log=_eviction_log(cluster_id, "plscience10.stsci.edu", failures=2),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True):
                reconcile_running_stages(state, "run_a", ctx)

            condor.write_bad_machines(artifacts["bad_machines"], {"plscience10.stsci.edu"})
            new_cluster = cluster_id + 1
            artifacts["log"].write_text(
                _eviction_log(new_cluster, "plscience11.stsci.edu", failures=2),
                encoding="utf-8",
            )
            state.try_atomic_claim(
                "run_a",
                label,
                "star",
                launch_token=state.new_launch_token(),
                executor="condor",
                native_id=new_cluster,
                log_path=str(
                    logs.target_log_path(str(tmp_path / "runs"), "run_a", label, "star")
                ),
                submit_epoch=time.time(),
            )
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={new_cluster: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True):
                counts = reconcile_running_stages(state, "run_a", ctx)

            bad = condor.read_bad_machines(artifacts["bad_machines"])
            self.assertEqual(
                bad,
                {"plscience10.stsci.edu", "plscience11.stsci.edu"},
            )
            merged = condor.apply_bad_machine_exclusions(
                condor.CondorResourceRequest(requirements="LoadAvg < 10"),
                artifacts,
            )
            self.assertIn("plscience10.stsci.edu", merged.requirements or "")
            self.assertIn("plscience11.stsci.edu", merged.requirements or "")
            self.assertEqual(counts["requeued"], 1)

    def test_diff_sidecar_activity_defers_eviction_requeue(self):
        cluster_id = 900_009
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = Target(
                sector=20,
                camera=3,
                ccd=3,
                target_ra=210.0,
                target_dec=81.0,
                target_name="2020ut",
            )
            state, run_dir = _minimal_condor_run(tmp_path, target)
            label = target.label()
            runs_root = tmp_path / "runs"
            log_path = logs.target_log_path(str(runs_root), "run_a", label, "diff")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("ePSF starting\n", encoding="utf-8")
            old = time.time() - 3600.0
            os.utime(log_path, (old, old))
            from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
                progress_path_for_diff_log,
            )

            sidecar = progress_path_for_diff_log(log_path)
            sidecar.write_text(
                json.dumps(
                    {
                        "phase": "running",
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "frames_done": 10,
                        "frames_total": 1188,
                    }
                ),
                encoding="utf-8",
            )
            artifacts = condor.condor_artifact_paths(str(runs_root), "run_a", label, "diff")
            artifacts["log"].write_text(
                _eviction_log(cluster_id, "plscience10.stsci.edu", failures=5),
                encoding="utf-8",
            )
            state.update_stage_status("run_a", label, "diff", STATUS_READY)
            state.try_atomic_claim(
                "run_a",
                label,
                "diff",
                launch_token=state.new_launch_token(),
                executor="condor",
                native_id=cluster_id,
                log_path=str(log_path),
                submit_epoch=time.time(),
            )
            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_IDLE, None)},
            ), unittest.mock.patch.object(condor, "remove_cluster") as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(counts["requeued"], 0)
            rm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
