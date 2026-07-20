"""Tests for Condor poll grace and RA normalization helpers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing.pancakes import (
    moc_ra_shift_degrees,
    normalize_ra_degrees,
    shift_polygon_ras_for_moc,
    shift_ras_for_moc,
)
from syndiff_pipeline.common.orchestration import condor, logs
from syndiff_pipeline.common.orchestration.run_context import resolve_run_context
from syndiff_pipeline.common.orchestration.scheduler import reconcile_running_stages
from syndiff_pipeline.common.orchestration.state import (
    PipelineState,
    STAGE_DEPS,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_RUNNING,
)
from syndiff_pipeline.common.orchestration.targets import Target


class TestCondorPollGrace(unittest.TestCase):
    def test_poll_returns_none_within_grace_when_missing(self):
        cluster_id = 999_001
        condor._submission_times[cluster_id] = time.time()
        with unittest.mock.patch.object(condor, "_query_queue", return_value=(None, None)):
            with unittest.mock.patch.object(condor, "_query_history", return_value=(None, None)):
                self.assertIsNone(condor.poll_cluster(cluster_id))

    def test_poll_returns_failure_after_grace_when_missing(self):
        cluster_id = 999_002
        condor._submission_times[cluster_id] = time.time() - condor.poll_grace_seconds() - 1.0
        with unittest.mock.patch.object(condor, "_query_queue", return_value=(None, None)):
            with unittest.mock.patch.object(condor, "_query_history", return_value=(None, None)):
                self.assertEqual(condor.poll_cluster(cluster_id), 1)

    def test_poll_removed_with_exit_zero_is_canceled_not_success(self):
        cluster_id = 999_003
        with unittest.mock.patch.object(
            condor, "_query_queue", return_value=(None, None)
        ), unittest.mock.patch.object(
            condor, "_query_history", return_value=(condor._JOB_REMOVED, 0)
        ):
            self.assertEqual(condor.poll_cluster(cluster_id), 143)


def _minimal_condor_run(tmp: Path, target: Target) -> tuple[PipelineState, str]:
    state_db = tmp / "state.sqlite"
    runs_root = tmp / "runs"
    run_id = "run_a"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "per_target").mkdir()
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data_root: /data",
                f"workspace_root: {tmp}",
                f"runs_root: {runs_root}",
                f"state_db_path: {state_db}",
                "skycell_wcs_csv: skycells.csv",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "targets.csv").write_text(
        "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
        f"{target.sector},{target.camera},{target.ccd},1,1,{target.target_name},true\n",
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    state = PipelineState(str(state_db))
    state.create_run(
        run_id,
        str(cfg_path),
        str(run_dir / "targets.csv"),
        str(runs_root),
        [target],
        ["ps1_process"],
    )
    return state, str(run_dir)


class TestCondorGraceAcrossRestart(unittest.TestCase):
    """Reconcile must use the DB-persisted wall-clock submit_epoch for the poll
    grace, because the in-process ``_submission_times`` map is empty after a
    daemon restart."""

    def _target(self) -> Target:
        return Target(
            sector=40,
            camera=1,
            ccd=1,
            target_ra=292.6,
            target_dec=35.7,
            target_name="2021udg",
        )

    def _claimed_condor_run(self, tmp: Path, cluster_id: int, submit_epoch: float):
        target = self._target()
        state, run_dir = _minimal_condor_run(tmp, target)
        label = target.label()
        # Simulate a restart: the in-process submission map has no record.
        condor._submission_times.pop(cluster_id, None)
        state.update_stage_status("run_a", label, "ps1_process", STATUS_READY)
        state.try_atomic_claim(
            "run_a",
            label,
            "ps1_process",
            launch_token=state.new_launch_token(),
            executor="condor",
            native_id=cluster_id,
            log_path=str(logs.target_log_path(str(tmp / "runs"), "run_a", label, "ps1_process")),
            submit_epoch=submit_epoch,
        )
        ctx = resolve_run_context(run_dir=run_dir)
        return state, ctx, label

    def test_briefly_missing_cluster_within_grace_not_failed(self):
        cluster_id = 888_001
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, label = self._claimed_condor_run(
                tmp_path, cluster_id, submit_epoch=time.time()
            )
            with unittest.mock.patch.object(
                condor, "query_clusters", return_value={cluster_id: (None, None)}
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(counts["failed"], 0)
            self.assertEqual(row.status, STATUS_RUNNING)

    def test_missing_cluster_past_grace_is_failed(self):
        from syndiff_pipeline.common.orchestration import stage_liveness

        cluster_id = 888_002
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            past = time.time() - condor.poll_grace_seconds() - 1.0
            state, ctx, label = self._claimed_condor_run(
                tmp_path, cluster_id, submit_epoch=past
            )
            with unittest.mock.patch.object(
                condor, "query_clusters", return_value={cluster_id: (None, None)}
            ):
                for _ in range(stage_liveness.CONDOR_POLL_MISS_FAIL_THRESHOLD - 1):
                    counts = reconcile_running_stages(state, "run_a", ctx)
                    self.assertEqual(counts["failed"], 0)
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(row.status, STATUS_FAILED)


class TestCondorRestartRobustness(unittest.TestCase):
    def _target(self) -> Target:
        return Target(
            sector=20,
            camera=3,
            ccd=3,
            target_ra=210.0,
            target_dec=81.0,
            target_name="2020ut",
        )

    def _claimed_diff_run(self, tmp: Path, cluster_id: int, *, submit_epoch: float):
        target = self._target()
        state, run_dir = _minimal_condor_run(tmp, target)
        label = target.label()
        runs_root = tmp / "runs"
        log_path = logs.target_log_path(str(runs_root), "run_a", label, "ps1_process")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("processing\n", encoding="utf-8")
        artifacts = condor.condor_artifact_paths(str(runs_root), "run_a", label, "ps1_process")
        artifacts["clusters"].write_text(f"{cluster_id}\n", encoding="utf-8")
        state.update_stage_status("run_a", label, "ps1_process", STATUS_READY)
        state.try_atomic_claim(
            "run_a",
            label,
            "ps1_process",
            launch_token=state.new_launch_token(),
            executor="condor",
            native_id=cluster_id,
            log_path=str(log_path),
            submit_epoch=submit_epoch,
        )
        ctx = resolve_run_context(run_dir=run_dir)
        return state, ctx, label, log_path

    def test_active_log_defers_missing_cluster_failure(self):
        cluster_id = 888_010
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            past = time.time() - condor.poll_grace_seconds() - 1.0
            state, ctx, label, log_path = self._claimed_diff_run(
                tmp_path, cluster_id, submit_epoch=past
            )
            os.utime(log_path, (time.time(), time.time()))
            with unittest.mock.patch.object(
                condor, "query_clusters", return_value={cluster_id: (None, None)}
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(counts["failed"], 0)
            self.assertEqual(row.status, STATUS_RUNNING)

    def test_missing_cluster_fails_after_poll_miss_threshold(self):
        from syndiff_pipeline.common.orchestration import stage_liveness

        cluster_id = 888_011
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            past = time.time() - condor.poll_grace_seconds() - 1.0
            state, ctx, label, log_path = self._claimed_diff_run(
                tmp_path, cluster_id, submit_epoch=past
            )
            old = time.time() - stage_liveness.STAGE_OUTPUT_ACTIVE_S - 10.0
            os.utime(log_path, (old, old))
            with unittest.mock.patch.object(
                condor, "query_clusters", return_value={cluster_id: (None, None)}
            ):
                for _ in range(stage_liveness.CONDOR_POLL_MISS_FAIL_THRESHOLD - 1):
                    counts = reconcile_running_stages(state, "run_a", ctx)
                    self.assertEqual(counts["failed"], 0)
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(row.status, STATUS_FAILED)

    def test_reconcile_re_adopts_cluster_id_from_artifacts(self):
        cluster_id = 888_012
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = self._target()
            state, run_dir = _minimal_condor_run(tmp_path, target)
            label = target.label()
            runs_root = tmp_path / "runs"
            log_path = logs.target_log_path(str(runs_root), "run_a", label, "ps1_process")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("processing\n", encoding="utf-8")
            artifacts = condor.condor_artifact_paths(str(runs_root), "run_a", label, "ps1_process")
            artifacts["clusters"].write_text(f"{cluster_id}\n", encoding="utf-8")
            state.update_stage_status("run_a", label, "ps1_process", STATUS_READY)
            token = state.new_launch_token()
            state.try_atomic_claim(
                "run_a",
                label,
                "ps1_process",
                launch_token=token,
                executor="condor",
                native_id=None,
                log_path=str(log_path),
                submit_epoch=time.time(),
            )
            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch.object(
                condor,
                "query_clusters",
                return_value={cluster_id: (condor._JOB_RUNNING, None)},
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["adopted"], 1)
            self.assertEqual(row.native_id, cluster_id)
            self.assertEqual(row.status, STATUS_RUNNING)


def _real_disconnect_cycle(cluster_id: int, host: str, hhmmss: str) -> str:
    """Build one disconnect->reconnect-failed->evicted cycle in the exact
    format HTCondor writes to the user log (verified against a real
    production log from plscience10.stsci.edu evictions)."""
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
            "...",
        ]
    )


class TestCondorEvictionExclusion(unittest.TestCase):
    _SAMPLE_LOG = (
        _real_disconnect_cycle(5, "plscience10.stsci.edu", "16:59:20")
        + "\n...\n"
        + _real_disconnect_cycle(5, "plscience10.stsci.edu", "17:00:20")
    )

    def test_tally_execution_eviction_failures(self):
        tallies = condor.tally_execution_eviction_failures(self._SAMPLE_LOG)
        self.assertEqual(tallies, {"plscience10.stsci.edu": 2})

    def test_merge_requirements_with_exclusions(self):
        merged = condor.merge_requirements_with_exclusions(
            "Memory >= 100000",
            {"plscience10.stsci.edu"},
        )
        self.assertEqual(
            merged,
            '(Memory >= 100000) && Machine != "plscience10.stsci.edu"',
        )

    def test_submit_applies_bad_machine_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            submit_path = Path(tmp) / "job.submit"
            artifacts = {
                "stdout": Path(tmp) / "out",
                "stderr": Path(tmp) / "err",
                "log": Path(tmp) / "log",
                "bad_machines": Path(tmp) / "bad.json",
            }
            condor.write_bad_machines(
                artifacts["bad_machines"],
                {"plscience10.stsci.edu"},
            )
            resources = condor.apply_bad_machine_exclusions(
                condor.CondorResourceRequest(requirements="LoadAvg < 10"),
                artifacts,
            )
            condor.write_submit_file(
                submit_path,
                ["echo", "hi"],
                artifacts,
                resources,
            )
            text = submit_path.read_text(encoding="utf-8")
            self.assertIn('Machine != "plscience10.stsci.edu"', text)
            self.assertIn("LoadAvg < 10", text)

    def test_reconcile_requeues_on_eviction_loop(self):
        cluster_id = 888_020
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = Target(
                sector=20,
                camera=3,
                ccd=2,
                target_ra=210.0,
                target_dec=81.0,
                target_name="s20_astrometry",
            )
            state, run_dir = _minimal_condor_run(tmp_path, target)
            label = target.label()
            runs_root = tmp_path / "runs"
            log_path = logs.target_log_path(str(runs_root), "run_a", label, "star")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Stale stage log so reconcile does not treat output as active.
            log_path.write_text("starting\n", encoding="utf-8")
            old = time.time() - 3600.0
            os.utime(log_path, (old, old))
            artifacts = condor.condor_artifact_paths(str(runs_root), "run_a", label, "star")
            sample_log = (
                _real_disconnect_cycle(cluster_id, "plscience10.stsci.edu", "16:59:20")
                + "\n...\n"
                + _real_disconnect_cycle(cluster_id, "plscience10.stsci.edu", "17:00:20")
            )
            artifacts["log"].write_text(sample_log, encoding="utf-8")
            state.update_stage_status("run_a", label, "star", STATUS_READY)
            state.try_atomic_claim(
                "run_a",
                label,
                "star",
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
            ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True) as rm:
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "star")
            self.assertEqual(counts["requeued"], 1)
            self.assertEqual(row.status, STATUS_READY)
            rm.assert_called_once()
            self.assertEqual(rm.call_args.args[0], cluster_id)
            bad = condor.read_bad_machines(artifacts["bad_machines"])
            self.assertIn("plscience10.stsci.edu", bad)


class TestCondorHoldTimeoutConfig(unittest.TestCase):
    def test_runner_config_defaults_hold_timeout(self):
        from syndiff_pipeline.template_creation.orchestration.runner_config import (
            RunnerConfig,
        )

        cfg = RunnerConfig()
        self.assertEqual(cfg.condor_hold_timeout_s, 600.0)

    def test_runner_config_loads_hold_timeout_from_yaml(self):
        from syndiff_pipeline.template_creation.orchestration.runner_config import (
            load_and_materialize_runner_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "data_root: /data",
                        f"workspace_root: {tmp}",
                        f"runs_root: {tmp}/runs",
                        f"state_db_path: {tmp}/state.sqlite",
                        "skycell_wcs_csv: skycells.csv",
                        "scheduler:",
                        "  condor_hold_timeout_s: 120.0",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_and_materialize_runner_config(cfg_path)
            self.assertEqual(cfg.condor_hold_timeout_s, 120.0)

    def test_reconcile_passes_configured_hold_timeout(self):
        cluster_id = 777_010
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = Target(
                sector=40,
                camera=1,
                ccd=1,
                target_ra=292.6,
                target_dec=35.7,
                target_name="2021udg",
            )
            state, run_dir = _minimal_condor_run(tmp_path, target)
            label = target.label()
            state.update_stage_status("run_a", label, "ps1_process", STATUS_READY)
            state.try_atomic_claim(
                "run_a",
                label,
                "ps1_process",
                launch_token=state.new_launch_token(),
                executor="condor",
                native_id=cluster_id,
                log_path=str(
                    logs.target_log_path(str(tmp_path / "runs"), "run_a", label, "ps1_process")
                ),
                submit_epoch=time.time(),
            )
            ctx = resolve_run_context(run_dir=run_dir)
            ctx.cfg.condor_hold_timeout_s = 42.0
            with unittest.mock.patch.object(
                condor, "query_clusters", return_value={cluster_id: (condor._JOB_HELD, None)}
            ), unittest.mock.patch.object(
                condor, "poll_cluster_status", return_value=None
            ) as poll:
                reconcile_running_stages(state, "run_a", ctx)
            poll.assert_called_once()
            self.assertEqual(poll.call_args.kwargs["hold_timeout_s"], 42.0)


class TestCondorHeldJob(unittest.TestCase):
    def setUp(self):
        condor._held_times.clear()

    def test_held_within_timeout_returns_none(self):
        cluster_id = 777_001
        condor._held_times[cluster_id] = time.time()
        with unittest.mock.patch.object(
            condor, "_query_hold_reason", return_value="Memory exceeded"
        ):
            self.assertIsNone(
                condor.poll_cluster_status(
                    cluster_id,
                    condor._JOB_HELD,
                    None,
                    submitted_at=time.time(),
                    hold_timeout_s=600.0,
                )
            )

    def test_held_past_timeout_removes_and_fails(self):
        cluster_id = 777_002
        condor._held_times[cluster_id] = time.time() - 601.0
        with unittest.mock.patch.object(
            condor, "_query_hold_reason", return_value="Memory exceeded"
        ), unittest.mock.patch.object(condor, "remove_cluster", return_value=True) as rm:
            exit_code = condor.poll_cluster_status(
                cluster_id,
                condor._JOB_HELD,
                None,
                submitted_at=time.time(),
                hold_timeout_s=600.0,
            )
        self.assertEqual(exit_code, 1)
        rm.assert_called_once_with(cluster_id)

    def test_hold_file_persists_first_held_epoch(self):
        cluster_id = 777_003
        with tempfile.TemporaryDirectory() as tmp:
            hold_path = Path(tmp) / "ps1_process.condor.hold"
            now = time.time()
            with unittest.mock.patch.object(
                condor, "_query_hold_reason", return_value="Memory exceeded"
            ):
                self.assertIsNone(
                    condor.poll_cluster_status(
                        cluster_id,
                        condor._JOB_HELD,
                        None,
                        submitted_at=now,
                        hold_timeout_s=600.0,
                        hold_path=hold_path,
                    )
                )
            self.assertTrue(hold_path.is_file())
            persisted = float(hold_path.read_text(encoding="utf-8").strip())
            self.assertAlmostEqual(persisted, now, places=3)

    def test_hold_file_survives_daemon_restart(self):
        cluster_id = 777_004
        with tempfile.TemporaryDirectory() as tmp:
            hold_path = Path(tmp) / "ps1_process.condor.hold"
            held_since = time.time() - 601.0
            hold_path.write_text(f"{held_since}\n", encoding="utf-8")
            condor._held_times.clear()
            with unittest.mock.patch.object(
                condor, "_query_hold_reason", return_value="Memory exceeded"
            ), unittest.mock.patch.object(
                condor, "remove_cluster", return_value=True
            ) as rm:
                exit_code = condor.poll_cluster_status(
                    cluster_id,
                    condor._JOB_HELD,
                    None,
                    submitted_at=time.time(),
                    hold_timeout_s=600.0,
                    hold_path=hold_path,
                )
            self.assertEqual(exit_code, 1)
            rm.assert_called_once_with(cluster_id)
            self.assertFalse(hold_path.exists())

    def test_hold_file_cleared_on_completion(self):
        cluster_id = 777_005
        with tempfile.TemporaryDirectory() as tmp:
            hold_path = Path(tmp) / "ps1_process.condor.hold"
            hold_path.write_text(f"{time.time()}\n", encoding="utf-8")
            condor.poll_cluster_status(
                cluster_id,
                condor._JOB_COMPLETED,
                0,
                hold_path=hold_path,
            )
            self.assertFalse(hold_path.exists())


class TestQueryHistoryLimit(unittest.TestCase):
    def test_query_history_uses_limit(self):
        with unittest.mock.patch.object(condor, "_run_condor") as run:
            run.return_value = unittest.mock.Mock(stdout="", stderr="", returncode=0)
            condor._query_history(12345)
            args = run.call_args[0][0]
            self.assertIn("-limit", args)
            self.assertIn("1", args)


class TestQueryClustersBatch(unittest.TestCase):
    def test_query_clusters_parses_batched_queue_and_history_fallback(self):
        with unittest.mock.patch.object(condor, "_run_condor") as run, unittest.mock.patch.object(
            condor, "_query_history", side_effect=[(condor._JOB_COMPLETED, 0), (None, None)]
        ) as history:
            run.return_value = unittest.mock.Mock(
                stdout="100001 2 undefined\n100003 2 undefined\n",
                stderr="",
                returncode=0,
            )
            result = condor.query_clusters([100001, 100002, 100003])
        self.assertEqual(result[100001], (2, None))
        self.assertEqual(result[100003], (2, None))
        self.assertEqual(result[100002], (condor._JOB_COMPLETED, 0))
        history.assert_called_once_with(100002)


class TestGarbledCondorOutput(unittest.TestCase):
    def test_garbled_queue_output_does_not_raise(self):
        with unittest.mock.patch.object(condor, "_run_condor") as run, unittest.mock.patch.object(
            condor, "_query_history", return_value=(None, None)
        ):
            run.return_value = unittest.mock.Mock(
                stdout="garbage not-a-number\n",
                stderr="",
                returncode=0,
            )
            result = condor.query_clusters([999_999])
        self.assertEqual(result[999_999], (None, None))


class TestWriteSubmitFileEnvironment(unittest.TestCase):
    def test_includes_environment_when_conda_sh_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            submit_path = Path(tmp) / "job.submit"
            artifacts = {
                "stdout": Path(tmp) / "out",
                "stderr": Path(tmp) / "err",
                "log": Path(tmp) / "log",
            }
            with unittest.mock.patch.dict(
                os.environ,
                {"SYNDIFF_CONDA_SH": "/opt/conda/etc/profile.d/conda.sh"},
                clear=False,
            ):
                condor.write_submit_file(
                    submit_path,
                    ["echo", "hi"],
                    artifacts,
                    condor.CondorResourceRequest(),
                )
            text = submit_path.read_text(encoding="utf-8")
            self.assertIn('environment = "SYNDIFF_CONDA_SH=', text)
            self.assertIn("SYNDIFF_CONDA_ENV=", text)

    def test_omits_environment_when_conda_sh_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            submit_path = Path(tmp) / "job.submit"
            artifacts = {
                "stdout": Path(tmp) / "out",
                "stderr": Path(tmp) / "err",
                "log": Path(tmp) / "log",
            }
            env = os.environ.copy()
            env.pop("SYNDIFF_CONDA_SH", None)
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                condor.write_submit_file(
                    submit_path,
                    ["echo", "hi"],
                    artifacts,
                    condor.CondorResourceRequest(),
                )
            text = submit_path.read_text(encoding="utf-8")
            self.assertNotIn("SYNDIFF_CONDA_SH=", text)
            self.assertNotIn("SYNDIFF_CONDA_ENV=", text)


class TestStageDeps(unittest.TestCase):
    def test_downsample_requires_mapping_ps1_and_remap(self):
        self.assertEqual(STAGE_DEPS["downsample"], ["mapping", "ps1_process", "remap"])


class TestRaNormalization(unittest.TestCase):
    def test_normalize_ra_degrees_wraps(self):
        ra = normalize_ra_degrees(np.array([-10.0, 370.0, 358.0]))
        np.testing.assert_allclose(ra, [350.0, 10.0, 358.0])

    def test_shift_polygon_ras_for_moc_spans_zero(self):
        vertices = np.array(
            [
                [[359.0, 0.0], [1.0, 0.0], [1.0, 1.0], [359.0, 1.0]],
            ],
            dtype=np.float64,
        )
        shift = moc_ra_shift_degrees(358.0)
        out = shift_polygon_ras_for_moc(vertices, shift)
        self.assertTrue(np.all(out[:, :, 0] >= 0.0))
        self.assertTrue(np.all(out[:, :, 0] < 360.0))
        self.assertLess(out[0, :, 0].max() - out[0, :, 0].min(), 180.0)

    def test_shift_ras_for_moc_near_zero(self):
        shift = moc_ra_shift_degrees(358.0)
        ra = shift_ras_for_moc(np.array([359.0, 0.5, 1.0]), shift)
        self.assertTrue(np.all(ra >= 0.0))
        self.assertTrue(np.all(ra < 360.0))


if __name__ == "__main__":
    unittest.main()
