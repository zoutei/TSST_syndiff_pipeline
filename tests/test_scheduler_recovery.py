"""Tests for supervisor reconcile and daemon lifecycle."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner import logs
from syndiff_pipeline.template_runner.launcher import LaunchDescriptor
from syndiff_pipeline.template_runner.scheduler import (
    _LOCAL_START_GRACE_S,
    _try_launch_ready_row,
    reconcile_running_stages,
)
from syndiff_pipeline.template_runner.scheduler_control import daemon_is_alive
from syndiff_pipeline.template_runner.state import (
    PipelineState,
    STATUS_CANCELED,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SUCCESS,
)
from syndiff_pipeline.template_runner.targets import Target


def _minimal_run(tmp: Path, target: Target, stages: list[str]) -> tuple[PipelineState, str]:
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
                f"handoff_root: {tmp}",
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
        stages,
    )
    return state, str(run_dir)


class TestSchedulerReconcile(unittest.TestCase):
    def _target(self) -> Target:
        return Target(
            sector=40,
            camera=1,
            ccd=1,
            target_ra=292.646875,
            target_dec=35.776111,
            target_name="2021udg",
        )

    def test_reconcile_requeues_dead_local_running_stage(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
            label = target.label()
            token = state.new_launch_token()
            state.update_stage_status("run_a", label, "ps1_download", STATUS_READY)
            state.try_atomic_claim(
                "run_a",
                label,
                "ps1_download",
                launch_token=token,
                executor="local",
                native_id=999999,
                log_path=str(tmp_path / "x.log"),
            )
            from syndiff_pipeline.template_runner.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_download")
            self.assertEqual(counts["requeued"], 1)
            self.assertEqual(row.status, STATUS_READY)

    def test_reconcile_adopts_live_local_pid_without_duplicate(self):
        # A running row whose status.json launch_token matches and whose pid is
        # a live local process must be ADOPTED (left 'running'), never requeued
        # or relaunched/duplicated.
        target = self._target()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
                label = target.label()
                runs_root = str(tmp_path / "runs")
                token = state.new_launch_token()
                state.update_stage_status("run_a", label, "ps1_download", STATUS_READY)
                state.try_atomic_claim(
                    "run_a",
                    label,
                    "ps1_download",
                    launch_token=token,
                    executor="local",
                    native_id=proc.pid,
                    log_path=str(
                        logs.target_log_path(runs_root, "run_a", label, "ps1_download")
                    ),
                )
                # Worker wrote its status file but is still running (not exited).
                logs.write_json_atomic(
                    logs.stage_status_path(runs_root, "run_a", label, "ps1_download"),
                    {
                        "launch_token": token,
                        "pid": proc.pid,
                        "state": "running",
                    },
                )
                from syndiff_pipeline.template_runner.run_context import resolve_run_context

                ctx = resolve_run_context(run_dir=run_dir)
                counts = reconcile_running_stages(state, "run_a", ctx)
                row = state.get_stage_run("run_a", label, "ps1_download")
                self.assertEqual(counts["adopted"], 1)
                self.assertEqual(counts["requeued"], 0)
                self.assertEqual(counts["still_running"], 1)
                self.assertEqual(row.status, STATUS_RUNNING)
                # The launch token is preserved (not relaunched with a new one).
                self.assertEqual(row.launch_token, token)
                self.assertEqual(row.native_id, proc.pid)
        finally:
            proc.kill()

    def test_reconcile_finalizes_exited_status_file(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
            label = target.label()
            token = state.new_launch_token()
            runs_root = str(tmp_path / "runs")
            state.update_stage_status("run_a", label, "ps1_download", STATUS_READY)
            state.try_atomic_claim(
                "run_a",
                label,
                "ps1_download",
                launch_token=token,
                executor="local",
                native_id=999999,
                log_path=str(logs.target_log_path(runs_root, "run_a", label, "ps1_download")),
            )
            logs.write_json_atomic(
                logs.stage_status_path(runs_root, "run_a", label, "ps1_download"),
                {
                    "launch_token": token,
                    "pid": 999999,
                    "state": "success",
                    "exit_code": 0,
                },
            )
            from syndiff_pipeline.template_runner.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_download")
            self.assertEqual(counts["completed"], 1)
            self.assertEqual(row.status, STATUS_SUCCESS)

    def test_reconcile_marks_finished_condor_stage_success(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, run_dir = _minimal_run(tmp_path, target, ["ps1_process"])
            label = target.label()
            state.update_stage_status("run_a", label, "ps1_process", STATUS_READY)
            state.try_atomic_claim(
                "run_a",
                label,
                "ps1_process",
                launch_token=state.new_launch_token(),
                executor="condor",
                native_id=54,
                log_path=str(tmp_path / "x.log"),
                submit_epoch=1_700_000_000.0,
            )
            from syndiff_pipeline.template_runner.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler.condor.poll_cluster",
                return_value=0,
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_process")
            self.assertEqual(counts["completed"], 1)
            self.assertEqual(row.status, STATUS_SUCCESS)

    def test_reconcile_marks_condor_removed_job_canceled(self):
        """condor_rm with clean SIGTERM exit must not finalize as success."""
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, run_dir = _minimal_run(tmp_path, target, ["mapping"])
            label = target.label()
            runs_root = str(tmp_path / "runs")
            log_path = logs.target_log_path(runs_root, "run_a", label, "mapping")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "INFO Received signal 15. Initiating graceful shutdown...\n"
                "INFO Graceful shutdown initiated. Exiting...\n",
                encoding="utf-8",
            )
            state.update_stage_status("run_a", label, "mapping", STATUS_READY)
            state.try_atomic_claim(
                "run_a",
                label,
                "mapping",
                launch_token=state.new_launch_token(),
                executor="condor",
                native_id=56,
                log_path=str(log_path),
                submit_epoch=1_700_000_000.0,
            )
            from syndiff_pipeline.template_runner.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler.condor.poll_cluster",
                return_value=143,
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "mapping")
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(row.status, STATUS_CANCELED)
            self.assertEqual(row.exit_code, 143)

    def test_reconcile_grace_when_alive_with_stale_status_token(self):
        """Alive child with a previous launch's token in status.json must not requeue."""
        target = self._target()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
                label = target.label()
                runs_root = str(tmp_path / "runs")
                stale_token = state.new_launch_token()
                current_token = state.new_launch_token()
                state.update_stage_status("run_a", label, "ps1_download", STATUS_READY)
                state.try_atomic_claim(
                    "run_a",
                    label,
                    "ps1_download",
                    launch_token=current_token,
                    executor="local",
                    native_id=proc.pid,
                    log_path=str(
                        logs.target_log_path(runs_root, "run_a", label, "ps1_download")
                    ),
                )
                logs.write_json_atomic(
                    logs.stage_status_path(runs_root, "run_a", label, "ps1_download"),
                    {
                        "launch_token": stale_token,
                        "pid": proc.pid,
                        "state": "running",
                    },
                )
                from syndiff_pipeline.template_runner.run_context import resolve_run_context

                ctx = resolve_run_context(run_dir=run_dir)
                counts = reconcile_running_stages(state, "run_a", ctx)
                row = state.get_stage_run("run_a", label, "ps1_download")
                self.assertEqual(counts["still_running"], 1)
                self.assertEqual(counts["requeued"], 0)
                self.assertEqual(row.status, STATUS_RUNNING)
                self.assertEqual(row.launch_token, current_token)
        finally:
            proc.kill()

    def test_reconcile_requeues_stale_token_after_grace_and_terminates(self):
        target = self._target()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
                label = target.label()
                runs_root = str(tmp_path / "runs")
                stale_token = state.new_launch_token()
                current_token = state.new_launch_token()
                state.update_stage_status("run_a", label, "ps1_download", STATUS_READY)
                state.try_atomic_claim(
                    "run_a",
                    label,
                    "ps1_download",
                    launch_token=current_token,
                    executor="local",
                    native_id=proc.pid,
                    log_path=str(
                        logs.target_log_path(runs_root, "run_a", label, "ps1_download")
                    ),
                )
                logs.write_json_atomic(
                    logs.stage_status_path(runs_root, "run_a", label, "ps1_download"),
                    {
                        "launch_token": stale_token,
                        "pid": proc.pid,
                        "state": "running",
                    },
                )
                from syndiff_pipeline.template_runner.run_context import resolve_run_context

                ctx = resolve_run_context(run_dir=run_dir)
                with unittest.mock.patch(
                    "syndiff_pipeline.template_runner.scheduler._age_seconds",
                    return_value=_LOCAL_START_GRACE_S + 1,
                ), unittest.mock.patch(
                    "syndiff_pipeline.template_runner.scheduler._terminate_job",
                ) as mock_terminate:
                    counts = reconcile_running_stages(state, "run_a", ctx)
                row = state.get_stage_run("run_a", label, "ps1_download")
                self.assertEqual(counts["requeued"], 1)
                self.assertEqual(row.status, STATUS_READY)
                mock_terminate.assert_called_once()
        finally:
            proc.kill()

    def test_reconcile_stale_token_does_not_duplicate_on_tick(self):
        target = self._target()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
                label = target.label()
                runs_root = str(tmp_path / "runs")
                stale_token = state.new_launch_token()
                current_token = state.new_launch_token()
                state.update_stage_status("run_a", label, "ps1_download", STATUS_READY)
                state.try_atomic_claim(
                    "run_a",
                    label,
                    "ps1_download",
                    launch_token=current_token,
                    executor="local",
                    native_id=proc.pid,
                    log_path=str(
                        logs.target_log_path(runs_root, "run_a", label, "ps1_download")
                    ),
                )
                logs.write_json_atomic(
                    logs.stage_status_path(runs_root, "run_a", label, "ps1_download"),
                    {
                        "launch_token": stale_token,
                        "pid": proc.pid,
                        "state": "running",
                    },
                )
                from syndiff_pipeline.template_runner.run_context import resolve_run_context

                ctx = resolve_run_context(run_dir=run_dir)
                for _ in range(2):
                    counts = reconcile_running_stages(state, "run_a", ctx)
                    self.assertEqual(counts["requeued"], 0)
                    self.assertEqual(counts["still_running"], 1)
                row = state.get_stage_run("run_a", label, "ps1_download")
                self.assertEqual(row.status, STATUS_RUNNING)
                self.assertEqual(row.launch_token, current_token)
        finally:
            proc.kill()

    def test_launch_clears_stale_status_file(self):
        target = self._target()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, run_dir = _minimal_run(tmp_path, target, ["mapping", "ps1_download"])
            label = target.label()
            runs_root = str(tmp_path / "runs")
            state.update_stage_status(
                "run_a", label, "mapping", STATUS_SUCCESS, exit_code=0
            )
            state.cache_external_check("run_a", label, "ps1_download", complete=True)
            state.update_stage_status("run_a", label, "ps1_download", STATUS_READY)
            status_path = logs.stage_status_path(runs_root, "run_a", label, "ps1_download")
            logs.write_json_atomic(
                status_path,
                {"launch_token": "old-token", "pid": 1, "state": "running"},
            )
            row = state.get_stage_run("run_a", label, "ps1_download")
            from syndiff_pipeline.template_runner.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            seen_at_launch: list[bool] = []

            def fake_launch(*args, **kwargs):
                seen_at_launch.append(status_path.is_file())
                return LaunchDescriptor(
                    executor="local",
                    native_id=424242,
                    launch_token=kwargs["launch_token"],
                    submit_epoch=1.0,
                )

            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler.launcher.launch_stage",
                side_effect=fake_launch,
            ):
                launched = _try_launch_ready_row(
                    state,
                    "run_a",
                    ctx,
                    row,
                    pool_label="network",
                    force_rerun=False,
                    active_stages=["ps1_download"],
                    targets_by_label={label: target},
                    runs_root=runs_root,
                )
            self.assertTrue(launched)
            self.assertEqual(seen_at_launch, [False])


class TestDaemonIsAlive(unittest.TestCase):
    def test_stale_supervisor_heartbeat_is_not_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            state = PipelineState(db)
            state.update_supervisor_heartbeat(12345)
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.read_pid",
                return_value=999999,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.is_process_alive",
                return_value=False,
            ):
                with state._conn() as conn:
                    conn.execute(
                        "UPDATE daemon SET last_heartbeat = ? WHERE id = 1",
                        ("2020-01-01T00:00:00+00:00",),
                    )
                self.assertFalse(daemon_is_alive(db))

    def test_fresh_local_heartbeat_overrides_stale_db_heartbeat(self):
        """A busy daemon (fresh local heartbeat) must read as alive / not wedged,
        even when the NFS DB heartbeat is stale or unwritable. This is the core
        of the wedge fix: liveness comes from the host-local file, not the DB."""
        from syndiff_pipeline.template_runner.scheduler import _write_local_heartbeat
        from syndiff_pipeline.template_runner.scheduler_control import daemon_is_wedged

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            state = PipelineState(db)
            state.update_supervisor_heartbeat(4321)
            # Make the DB heartbeat ancient (simulating a wedged/slow NFS write).
            with state._conn() as conn:
                conn.execute(
                    "UPDATE daemon SET last_heartbeat = ? WHERE id = 1",
                    ("2020-01-01T00:00:00+00:00",),
                )
            # Heartbeat thread keeps the LOCAL file fresh.
            _write_local_heartbeat(db)
            self.addCleanup(
                lambda: logs.daemon_heartbeat_file(db).unlink(missing_ok=True)
            )
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.read_pid",
                return_value=4321,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.is_process_alive",
                return_value=True,
            ):
                self.assertTrue(daemon_is_alive(db))
                self.assertFalse(daemon_is_wedged(db))

    def test_alive_pid_with_no_heartbeat_is_wedged(self):
        """A live pid with no fresh heartbeat anywhere is the true wedge case."""
        from syndiff_pipeline.template_runner.scheduler_control import daemon_is_wedged

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            PipelineState(db)  # creates schema, no heartbeat row
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.read_pid",
                return_value=4321,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.is_process_alive",
                return_value=True,
            ):
                self.assertTrue(daemon_is_wedged(db))

    def test_fresh_local_heartbeat_without_live_pid_is_not_alive(self):
        """Ghost liveness after kill: local heartbeat fresh but pid gone."""
        from syndiff_pipeline.template_runner.scheduler import _write_local_heartbeat

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            _write_local_heartbeat(db)
            self.addCleanup(
                lambda: logs.daemon_heartbeat_file(db).unlink(missing_ok=True)
            )
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.read_pid",
                return_value=None,
            ):
                self.assertFalse(daemon_is_alive(db))


if __name__ == "__main__":
    unittest.main()
