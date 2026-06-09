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

from syndiff_pipeline.template_runner import logs
from syndiff_pipeline.template_runner.runner_config import load_runner_config, resolve_config
from syndiff_pipeline.template_runner.scheduler import (
    _apply_commands,
    _apply_verify_outcome,
    _iter_verify_candidates,
    _run_verify_pass,
    _tick_run,
)
from syndiff_pipeline.template_runner.state import (
    PipelineState,
    STATUS_EXTERNAL,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_SKIPPED,
)
from syndiff_pipeline.template_runner.targets import Target
from syndiff_pipeline.template_runner.verify import (
    copy_manifest_to_stable,
    manifest_valid,
    read_manifest,
    write_manifest,
)
from syndiff_pipeline.template_runner.verify_status import (
    clear_verify_in_flight,
    read_verify_in_flight,
    write_verify_in_flight,
)
from syndiff_pipeline.template_runner.verify_worker import (
    get_verify_worker,
    reset_verify_worker_for_tests,
    shutdown_verify_worker,
    try_get_verify_worker,
)
from tests.test_daemon_behavior import _minimal_run_setup, _write_mapping_csv_and_manifest


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
                        f"handoff_root: {tmp_path}",
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

    def test_read_missing_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_verify_in_flight(Path(tmp) / "nope.sqlite"), 0)


class TestVerifyWorkerLifecycle(unittest.TestCase):
    def setUp(self):
        reset_verify_worker_for_tests()

    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_try_get_returns_none_before_init(self):
        self.assertIsNone(try_get_verify_worker())

    def test_cancel_paths_do_not_create_worker(self):
        from syndiff_pipeline.template_runner.scheduler import _cancel_verify_run

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

            def slow_complete(*_args, **_kwargs):
                time.sleep(2.0)
                return False

            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.verify_worker.stage_complete",
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

    def test_async_verify_result_applied_on_later_pass(self):
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

            def slow_complete(*_args, **_kwargs):
                time.sleep(0.4)
                return False

            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.verify_worker.stage_complete",
                side_effect=slow_complete,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=False
                )
                self.assertFalse(state.external_checked(run_id, label, "mapping"))
                _run_verify_pass(
                    state,
                    run_id,
                    ctx,
                    force_rerun=False,
                    budget=16,
                    block=True,
                    block_timeout_s=3.0,
                )

            self.assertTrue(state.external_checked(run_id, label, "mapping"))
            promoted = state.promote_stages(run_id)
            self.assertEqual(promoted, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_READY
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

            def slow_complete(*_args, **_kwargs):
                time.sleep(0.8)
                return True

            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.verify_worker.stage_complete",
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

            def slow_complete(*_args, **_kwargs):
                time.sleep(1.0)
                return False

            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.verify_worker.stage_complete",
                side_effect=slow_complete,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.template_runner.scheduler.launcher.launch_stage",
            ):
                _tick_run(state, run_id, ctx)
                run = state.get_run(run_id)
                self.assertNotEqual(run.get("status"), "stalled")
                self.assertGreaterEqual(get_verify_worker().in_flight_count(run_id), 1)


if __name__ == "__main__":
    unittest.main()
