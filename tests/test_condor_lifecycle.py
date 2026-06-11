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
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(row.status, STATUS_FAILED)


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
            self.assertNotIn("environment =", text)


class TestStageDeps(unittest.TestCase):
    def test_downsample_requires_mapping(self):
        self.assertEqual(STAGE_DEPS["downsample"], ["wcs_grouping", "mapping", "ps1_process"])


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
