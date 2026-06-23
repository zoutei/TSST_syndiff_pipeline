"""Tests for async artifact verification and related helpers."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config, resolve_config
from syndiff_pipeline.common.orchestration.scheduler import (
    _apply_commands,
    _apply_verify_outcome,
    _iter_verify_candidates,
    _run_verify_pass,
    _tick_run,
    _verify_backlog,
)
from syndiff_pipeline.common.orchestration.state import (
    PipelineState,
    STATUS_EXTERNAL,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_SKIPPED,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.verify import (
    copy_manifest_to_stable,
    manifest_valid,
    read_manifest,
    write_manifest,
)
from syndiff_pipeline.common.orchestration.verify_status import (
    clear_verify_in_flight,
    read_verify_in_flight,
    read_verify_pending,
    read_verify_run_status,
    refresh_verify_run_status,
    write_verify_in_flight,
)
from syndiff_pipeline.common.orchestration.verify_worker import (
    get_verify_worker,
    reset_verify_worker_for_tests,
    shutdown_verify_worker,
    try_get_verify_worker,
)
from tests.test_daemon_behavior import _minimal_run_setup, _write_mapping_csv_and_manifest


def _ensure_mapping_csv_exists(ctx, target: Target) -> None:
    """Create a stub mapping CSV so absence probe defers to full verify."""
    resolved = resolve_config(target, ctx.cfg)
    csv_path = (
        Path(resolved.mapping_root)
        / f"sector_{target.sector:04d}"
        / f"camera_{target.camera}"
        / f"ccd_{target.ccd}"
        / f"tess_s{target.sector:04d}_{target.camera}_{target.ccd}_master_skycells_list.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("NAME,projection\nskycell.0001.0001,0001\n", encoding="utf-8")


def _mock_launch_descriptor(**kwargs):
    from syndiff_pipeline.common.orchestration.launcher import LaunchDescriptor

    return LaunchDescriptor(
        executor="local",
        native_id=12345,
        launch_token=str(kwargs.get("launch_token", "tok")),
        submit_epoch=time.time(),
        handle=None,
    )


class TestCopyManifestToStable(unittest.TestCase):
    def test_copies_valid_manifest_atomically(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_root = tmp_path / "runs"
            run_dir = runs_root / "run_a"
            run_dir.mkdir(parents=True)
            cfg_path = run_dir / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        f"data_root: {tmp_path / 'data'}",
                        f"workspace_root: {tmp_path}",
                        f"runs_root: {runs_root}",
                        f"state_db_path: {tmp_path / 'state.sqlite'}",
                        "skycell_wcs_csv: x.csv",
                    ]
                ),
                encoding="utf-8",
            )
            source = logs.stage_manifest_path(
                str(runs_root), "run_a", target.label(), "mapping"
            )
            stable = logs.stable_stage_manifest_path(
                str(runs_root), target.label(), "mapping"
            )
            _write_mapping_csv_and_manifest(tmp_path, target, source, runs_root=runs_root)

            self.assertTrue(copy_manifest_to_stable(source, stable))
            self.assertTrue(stable.is_file())
            copied = read_manifest(stable)
            self.assertIsNotNone(copied)
            cfg = load_runner_config(cfg_path)
            resolved = resolve_config(target, cfg)
            self.assertTrue(manifest_valid(copied, resolved, "mapping"))


class TestVerifyStatus(unittest.TestCase):
    def test_write_read_and_clear_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            write_verify_in_flight(db, {"run_a": 2, "run_b": 1})
            self.assertEqual(read_verify_in_flight(db), 3)
            self.assertEqual(read_verify_in_flight(db, "run_a"), 2)
            self.assertEqual(read_verify_in_flight(db, "run_missing"), 0)
            clear_verify_in_flight(db)
            self.assertEqual(read_verify_in_flight(db), 0)

    def test_extended_status_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            write_verify_in_flight(
                db,
                {
                    "run_a": {
                        "scan_running": 1,
                        "scan_queued": 4,
                        "active": [["s0069", "ps1_download"]],
                    }
                },
            )
            status = read_verify_run_status(db, "run_a")
            self.assertEqual(status["scan_running"], 1)
            self.assertEqual(status["scan_queued"], 4)
            self.assertEqual(status["active"], [["s0069", "ps1_download"]])
            self.assertEqual(read_verify_pending(db, "run_a"), 4)

    def test_read_missing_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_verify_in_flight(Path(tmp) / "nope.sqlite"), 0)

    def test_refresh_verify_run_status_noop_without_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            write_verify_in_flight(
                db,
                {
                    "run_a": {
                        "scan_running": 1,
                        "scan_queued": 0,
                        "active": [["s0020", "diff"]],
                    }
                },
            )
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.try_get_verify_worker",
                return_value=None,
            ):
                refresh_verify_run_status(db, unittest.mock.Mock(), "run_a")
            status = read_verify_run_status(db, "run_a")
            self.assertEqual(status["scan_running"], 1)

    def test_refresh_verify_run_status_updates_stale_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "state.sqlite")
            write_verify_in_flight(
                db,
                {
                    "run_a": {
                        "scan_running": 1,
                        "scan_queued": 2,
                        "active": [["s0020", "diff"]],
                    },
                    "run_b": {"scan_running": 3, "scan_queued": 0, "active": []},
                },
            )
            fresh = {"scan_running": 0, "scan_queued": 0, "active": []}
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.try_get_verify_worker",
                return_value=unittest.mock.Mock(),
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.collect_verify_status_for_run",
                return_value=fresh,
            ):
                refresh_verify_run_status(db, unittest.mock.Mock(), "run_a")
            status = read_verify_run_status(db, "run_a")
            self.assertEqual(status, fresh)
            self.assertEqual(read_verify_run_status(db, "run_b")["scan_running"], 3)


class TestVerifyWorkerLifecycle(unittest.TestCase):
    def setUp(self):
        reset_verify_worker_for_tests()

    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_try_get_returns_none_before_init(self):
        self.assertIsNone(try_get_verify_worker())

    def test_cancel_paths_do_not_create_worker(self):
        from syndiff_pipeline.common.orchestration.scheduler import _cancel_verify_run

        _cancel_verify_run("run_x")
        self.assertIsNone(try_get_verify_worker())


class TestVerifyScheduling(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_in_flight_cap_limits_parallel_scheduling(self):
        targets = [
            Target(22, 3, 3, 228.0, 52.0, "2020dgc"),
            Target(23, 1, 3, 185.0, 5.3, "2020ftl"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, targets, active_stages=["mapping"]
            )
            ctx.cfg.verify_max_workers = 1
            for target in targets:
                label = target.label()
                for stage in ("tess_ffi_download", "wcs_grouping"):
                    state.update_stage_status(
                        run_id, label, stage, STATUS_SKIPPED, exit_code=0
                    )
                    state.cache_external_check(run_id, label, stage, complete=True)
                _ensure_mapping_csv_exists(ctx, target)

            def slow_complete(*_args, **_kwargs):
                time.sleep(2.0)
                return False

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=False
                )

            worker = get_verify_worker()
            self.assertEqual(worker.in_flight_count(run_id), 1)
            shutdown_verify_worker(wait=False)

    def test_pending_candidates_sort_before_external(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)

            candidates = _iter_verify_candidates(
                state, run_id, ctx, force_rerun=False
            )
            statuses = {
                (key.target_label, key.stage): status for key, status, *_ in candidates
            }
            self.assertEqual(statuses[(label, "mapping")], STATUS_PENDING)
            if (label, "ps1_download") in statuses:
                self.assertEqual(statuses[(label, "ps1_download")], STATUS_EXTERNAL)
            mapping_idx = next(
                i for i, (key, _, *_) in enumerate(candidates) if key.stage == "mapping"
            )
            external_indices = [
                i
                for i, (key, status, *_) in enumerate(candidates)
                if status == STATUS_EXTERNAL
            ]
            if external_indices:
                self.assertLess(mapping_idx, min(external_indices))

    def test_incomplete_verify_promotes_and_stops_reverify(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)
            _ensure_mapping_csv_exists(ctx, target)

            def slow_complete(*_args, **_kwargs):
                time.sleep(0.4)
                return False

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=False
                )
                self.assertFalse(state.external_verify_complete(run_id, label, "mapping"))
                _run_verify_pass(
                    state,
                    run_id,
                    ctx,
                    force_rerun=False,
                    budget=16,
                    block=True,
                    block_timeout_s=3.0,
                )

            self.assertTrue(state.external_verify_attempted(run_id, label, "mapping"))
            self.assertFalse(state.external_verify_complete(run_id, label, "mapping"))
            promoted = state.promote_stages(run_id)
            self.assertEqual(promoted, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_READY
            )
            candidates = _iter_verify_candidates(
                state, run_id, ctx, force_rerun=False
            )
            self.assertFalse(any(key.stage == "mapping" for key, *_ in candidates))

    def test_verify_error_does_not_promote_stage(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)
            _ensure_mapping_csv_exists(ctx, target)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                side_effect=RuntimeError("disk read failed"),
            ):
                _run_verify_pass(
                    state,
                    run_id,
                    ctx,
                    force_rerun=False,
                    budget=16,
                    block=True,
                    block_timeout_s=3.0,
                )

            self.assertFalse(state.external_verify_attempted(run_id, label, "mapping"))
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_PENDING
            )
            candidates = _iter_verify_candidates(
                state, run_id, ctx, force_rerun=False
            )
            self.assertTrue(any(key.stage == "mapping" for key, *_ in candidates))
            promoted = state.promote_stages(run_id)
            self.assertEqual(promoted, 0)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_PENDING
            )


class TestVerifyCommandIntegration(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_force_rerun_command_cancels_entire_run_verify(self):
        import time

        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)
            _ensure_mapping_csv_exists(ctx, target)

            def slow_complete(*_args, **_kwargs):
                time.sleep(0.8)
                return True

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=False
                )
                self.assertGreaterEqual(get_verify_worker().in_flight_count(run_id), 1)
                state.insert_command(
                    "force_rerun",
                    run_id=run_id,
                    args={"target_labels": [label], "stages": ["mapping"]},
                )
                _apply_commands(state)
                get_verify_worker().drain(
                    lambda outcome: _apply_verify_outcome(state, outcome),
                    run_id=run_id,
                    block=True,
                    block_timeout_s=3.0,
                )

            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.status, STATUS_PENDING)

    def test_run_not_stalled_while_verify_in_flight(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)
            _ensure_mapping_csv_exists(ctx, target)

            def slow_complete(*_args, **_kwargs):
                time.sleep(1.0)
                return False

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=lambda *a, **kw: _mock_launch_descriptor(**kw),
            ):
                _tick_run(state, run_id, ctx)
                run = state.get_run(run_id)
                self.assertNotEqual(run.get("status"), "stalled")
                self.assertGreaterEqual(get_verify_worker().in_flight_count(run_id), 1)

    def test_run_not_stalled_with_verify_backlog(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)

            pending, in_flight = _verify_backlog(
                state, run_id, ctx, force_rerun=False
            )
            self.assertGreater(pending, 0)
            self.assertEqual(in_flight, 0)

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler._run_verify_pass",
                return_value=0,
            ):
                _tick_run(state, run_id, ctx)
                run = state.get_run(run_id)
                self.assertNotEqual(run.get("status"), "stalled")

    def test_stalled_run_resumes_when_verify_restarts(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)
            state.set_run_status(run_id, "stalled", stall_reason="test stall")
            _ensure_mapping_csv_exists(ctx, target)

            def slow_complete(*_args, **_kwargs):
                time.sleep(0.5)
                return False

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=lambda *a, **kw: _mock_launch_descriptor(**kw),
            ):
                _tick_run(state, run_id, ctx)
                run = state.get_run(run_id)
                self.assertEqual(run.get("status"), "running")
                self.assertEqual(run.get("stall_reason"), "")


if __name__ == "__main__":
    unittest.main()
