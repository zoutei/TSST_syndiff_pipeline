"""Behavior-level tests for supervisor daemon redesign."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner import daemon, logs
from syndiff_pipeline.template_runner.scheduler_control import stop_daemon
from syndiff_pipeline.template_runner.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_runner.scheduler import _resolve_external_and_pending_skips, _tick_run
from syndiff_pipeline.template_runner.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.template_runner.state import (
    PipelineState,
    STATUS_CANCELED,
    STATUS_EXTERNAL,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
)
from syndiff_pipeline.template_runner.targets import Target
from syndiff_pipeline.template_runner.verify import (
    persist_completion_manifests,
    stage_complete,
    stage_config_fingerprint,
    verify_mapping,
    write_manifest,
)


def _resolved(tmp: Path) -> ResolvedTargetConfig:
    target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
    return ResolvedTargetConfig(
        target=target,
        data_root=str(tmp / "data"),
        ffi_dir=str(tmp / "data" / "tess_ffi"),
        handoff_dir=str(tmp / "handoff" / target.label()),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        gaia_credentials=None,
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            downsample=DownsampleStageParams(single_offset=True),
        ),
        mapping_root=str(tmp / "mapping"),
        zarr_dir=str(tmp / "data" / "ps1_skycells_zarr"),
        template_output_base=str(tmp / "shifted_downsampled"),
    )


class TestAtomicClaim(unittest.TestCase):
    def test_only_one_claim_succeeds(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run("run_a", "/c", "/t", tmp, [target], ["mapping"])
            label = target.label()
            state.update_stage_status("run_a", label, "mapping", STATUS_READY)
            t1 = state.new_launch_token()
            t2 = state.new_launch_token()
            ok1 = state.try_atomic_claim(
                "run_a",
                label,
                "mapping",
                launch_token=t1,
                executor="local",
                native_id=100,
                log_path="/tmp/a.log",
            )
            ok2 = state.try_atomic_claim(
                "run_a",
                label,
                "mapping",
                launch_token=t2,
                executor="local",
                native_id=101,
                log_path="/tmp/b.log",
            )
            self.assertTrue(ok1)
            self.assertFalse(ok2)


class TestManifestFirstVerify(unittest.TestCase):
    def test_valid_manifest_marks_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            resolved = _resolved(tmp_path)
            manifest_path = tmp_path / "mapping.manifest.json"
            artifact = tmp_path / "artifact.fits"
            artifact.write_text("x", encoding="utf-8")
            write_manifest(
                logs.stage_manifest_path(
                    str(tmp_path), "run_a", resolved.target.label(), "mapping"
                ),
                resolved,
                "mapping",
                [str(artifact)],
                1,
                1,
            )
            path = logs.stage_manifest_path(
                str(tmp_path), "run_a", resolved.target.label(), "mapping"
            )
            self.assertTrue(stage_complete(resolved, "mapping", manifest_path=path))

    def test_stale_fingerprint_invalidates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            resolved = _resolved(tmp_path)
            artifact = tmp_path / "artifact.fits"
            artifact.write_text("x", encoding="utf-8")
            path = logs.stage_manifest_path(
                str(tmp_path), "run_a", resolved.target.label(), "mapping"
            )
            logs.write_json_atomic(
                path,
                {
                    "schema_version": 1,
                    "stage": "mapping",
                    "config_fingerprint": "stale",
                    "expected_count": 1,
                    "produced_count": 1,
                    "artifacts": [str(artifact)],
                },
            )
            self.assertFalse(stage_complete(resolved, "mapping", manifest_path=path))
            self.assertNotEqual(
                stage_config_fingerprint(resolved, "mapping"),
                "stale",
            )

    def test_stable_manifest_used_across_runs(self):
        # A stable (cross-run) manifest must satisfy stage_complete even when no
        # per-run manifest exists, so a fresh run skips re-scanning the output.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            resolved = _resolved(tmp_path)
            artifact = tmp_path / "artifact.fits"
            artifact.write_text("x", encoding="utf-8")
            stable_path = logs.stable_stage_manifest_path(
                str(tmp_path), resolved.target.label(), "mapping"
            )
            write_manifest(stable_path, resolved, "mapping", [str(artifact)], 1, 1)
            # No per-run manifest passed; stable path alone should mark complete.
            self.assertTrue(
                stage_complete(
                    resolved, "mapping", stable_manifest_path=str(stable_path)
                )
            )
            # If the recorded artifact disappears, the manifest is invalidated.
            artifact.unlink()
            self.assertFalse(
                stage_complete(
                    resolved, "mapping", stable_manifest_path=str(stable_path)
                )
            )

    def test_persist_completion_manifests_after_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            resolved = _resolved(tmp_path)
            csv_path = (
                Path(resolved.mapping_root)
                / "sector_0022"
                / "camera_3"
                / "ccd_3"
                / "tess_s0022_3_3_master_skycells_list.csv"
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("NAME,projection\nskycell.0001.0001,0001\n", encoding="utf-8")

            result = verify_mapping(resolved)
            self.assertTrue(result.ok)

            label = resolved.target.label()
            stable_path = logs.stable_stage_manifest_path(str(tmp_path), label, "mapping")
            per_run_path = logs.stage_manifest_path(str(tmp_path), "run_a", label, "mapping")
            written = persist_completion_manifests(
                resolved, "mapping", [per_run_path, stable_path]
            )
            self.assertEqual(written, [str(per_run_path), str(stable_path)])
            self.assertTrue(
                stage_complete(
                    resolved,
                    "mapping",
                    manifest_path=str(per_run_path),
                    stable_manifest_path=str(stable_path),
                )
            )


class TestSkipIntegration(unittest.TestCase):
    def test_external_stage_marked_skipped_when_complete(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_root = tmp_path / "runs"
            run_id = "run_a"
            run_dir = runs_root / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "per_target").mkdir()
            cfg_path = run_dir / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        f"data_root: {tmp_path / 'data'}",
                        f"handoff_root: {tmp_path}",
                        f"runs_root: {runs_root}",
                        f"state_db_path: {tmp_path / 'state.sqlite'}",
                        "skycell_wcs_csv: x.csv",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "targets.csv").write_text(
                "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
                "22,3,3,228.0,52.0,2020dgc,true\n",
                encoding="utf-8",
            )
            (run_dir / "run_meta.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            state = PipelineState(str(tmp_path / "state.sqlite"))
            state.create_run(
                run_id,
                str(cfg_path),
                str(run_dir / "targets.csv"),
                str(runs_root),
                [target],
                ["downsample"],
            )
            from syndiff_pipeline.template_runner.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            label = target.label()
            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.status, STATUS_EXTERNAL)
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler.stage_complete",
                return_value=True,
            ):
                skipped = _resolve_external_and_pending_skips(
                    state, run_id, ctx, force_rerun=False
                )
            self.assertGreaterEqual(skipped, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_SKIPPED
            )


class TestStallDetection(unittest.TestCase):
    def test_stalled_when_no_running_or_launchable(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_root = tmp_path / "runs"
            run_id = "run_a"
            run_dir = runs_root / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "per_target").mkdir()
            cfg_path = run_dir / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        f"data_root: {tmp_path / 'data'}",
                        f"handoff_root: {tmp_path}",
                        f"runs_root: {runs_root}",
                        f"state_db_path: {tmp_path / 'state.sqlite'}",
                        "skycell_wcs_csv: x.csv",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "targets.csv").write_text(
                "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
                "22,3,3,228.0,52.0,2020dgc,true\n",
                encoding="utf-8",
            )
            (run_dir / "run_meta.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            state = PipelineState(str(tmp_path / "state.sqlite"))
            state.create_run(
                run_id,
                str(cfg_path),
                str(run_dir / "targets.csv"),
                str(runs_root),
                [target],
                ["downsample"],
            )
            label = target.label()
            state.update_stage_status(run_id, label, "downsample", STATUS_PENDING)
            from syndiff_pipeline.template_runner.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler._resolve_external_and_pending_skips",
                return_value=0,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler.reconcile_running_stages",
                return_value={},
            ):
                _tick_run(state, run_id, ctx)
            run = state.get_run(run_id)
            self.assertEqual(run["status"], "stalled")
            self.assertIn("waiting on", run["stall_reason"])


class TestDaemonFlock(unittest.TestCase):
    def test_second_nonblocking_lock_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            with daemon.daemon_lock(db, blocking=True) as fd1:
                self.assertIsNotNone(fd1)
                with daemon.daemon_lock(db, blocking=False) as fd2:
                    self.assertIsNone(fd2)


class TestStopDaemon(unittest.TestCase):
    def test_cleans_stale_pid_file_when_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            pid_path = logs.daemon_pid_path(db)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("424242", encoding="utf-8")
            result = stop_daemon(db)
            self.assertFalse(result.was_running)
            self.assertTrue(result.stopped)
            self.assertFalse(result.force_killed)
            self.assertFalse(pid_path.is_file())

    def test_waits_for_graceful_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            pid_path = logs.daemon_pid_path(db)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("55555", encoding="utf-8")
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.is_process_alive",
                side_effect=[True, False],
            ) as alive, unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.terminate_process_tree",
            ) as term, unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.wait_for_process_exit",
                return_value=True,
            ) as wait:
                result = stop_daemon(db, term_timeout_s=0.1, kill_wait_s=0.1)
            self.assertTrue(result.was_running)
            self.assertTrue(result.stopped)
            self.assertFalse(result.force_killed)
            term.assert_called_once_with(55555, unittest.mock.ANY)
            wait.assert_called_once_with(55555, timeout_s=0.1)
            alive.assert_called()
            self.assertFalse(pid_path.is_file())

    def test_escalates_to_sigkill_when_term_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            pid_path = logs.daemon_pid_path(db)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("66666", encoding="utf-8")
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.is_process_alive",
                side_effect=[True, False],
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.terminate_process_tree",
            ) as term, unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.wait_for_process_exit",
                side_effect=[False, True],
            ) as wait:
                result = stop_daemon(db, term_timeout_s=0.1, kill_wait_s=0.1)
            self.assertTrue(result.force_killed)
            self.assertTrue(result.stopped)
            self.assertEqual(term.call_count, 2)
            term.assert_any_call(66666, unittest.mock.ANY)
            self.assertEqual(wait.call_count, 2)
            self.assertFalse(pid_path.is_file())

    def test_reports_failure_when_process_survives_sigkill(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            pid_path = logs.daemon_pid_path(db)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("77777", encoding="utf-8")
            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.is_process_alive",
                return_value=True,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.terminate_process_tree",
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler_control.daemon.wait_for_process_exit",
                return_value=False,
            ):
                result = stop_daemon(db, term_timeout_s=0.1, kill_wait_s=0.1)
            self.assertTrue(result.was_running)
            self.assertFalse(result.stopped)
            self.assertTrue(result.force_killed)
            self.assertTrue(pid_path.is_file())


class TestRetryAfterCancel(unittest.TestCase):
    def test_retry_reopens_canceled_stages(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run("run_a", "/c", "/t", tmp, [target], ["mapping", "downsample"])
            label = target.label()
            state.apply_cancel_run("run_a")
            self.assertEqual(
                state.get_stage_run("run_a", label, "mapping").status,
                STATUS_CANCELED,
            )
            state.apply_retry_run("run_a")
            self.assertEqual(state.get_stage_run("run_a", label, "mapping").status, STATUS_PENDING)


if __name__ == "__main__":
    unittest.main()
