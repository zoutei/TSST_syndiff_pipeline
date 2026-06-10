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

from syndiff_pipeline.template_creation.orchestration import daemon, logs
from syndiff_pipeline.template_creation.orchestration.scheduler import _write_local_heartbeat
from syndiff_pipeline.template_creation.orchestration.scheduler_control import daemon_is_alive, stop_daemon
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.scheduler import (
    _apply_verify_outcome,
    _cancel_verify_for_retry,
    _iter_verify_candidates,
    _resolve_external_and_pending_skips,
    _run_verify_pass,
    _tick_run,
)
from syndiff_pipeline.template_creation.orchestration.verify_worker import (
    VerifyTaskKey,
    get_verify_worker,
    reset_verify_worker_for_tests,
    shutdown_verify_worker,
)
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.template_creation.orchestration.run_report import format_status_grid
from syndiff_pipeline.template_creation.orchestration.state import (
    PipelineState,
    RUN_CANCELED,
    RUN_SUCCESS,
    SKIP_REASON_ARTIFACTS,
    SKIP_REASON_NOT_SELECTED,
    SKIP_REASON_SUPERSEDED,
    STATUS_CANCELED,
    STATUS_EXTERNAL,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    derive_run_final_status,
    upstream_stages_for,
)
from syndiff_pipeline.template_creation.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.verify import (
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
    def tearDown(self):
        reset_verify_worker_for_tests()

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
            from syndiff_pipeline.template_creation.orchestration.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            label = target.label()
            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.status, STATUS_EXTERNAL)
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
                return_value=True,
            ):
                skipped = _resolve_external_and_pending_skips(
                    state, run_id, ctx, force_rerun=False
                )
            self.assertGreaterEqual(skipped, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_SKIPPED
            )

    def test_force_rerun_verifies_external_prereqs_for_partial_run(self):
        target = Target(20, 3, 3, 210.219333, 81.846589, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path,
                [target],
                active_stages=["wcs_grouping", "downsample"],
                force_rerun=True,
            )
            label = target.label()

            def complete(_resolved, stage, **_kwargs):
                return stage in (
                    "tess_ffi_download",
                    "mapping",
                    "ps1_download",
                    "ps1_process",
                )

            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
                side_effect=complete,
            ):
                skipped = _resolve_external_and_pending_skips(
                    state, run_id, ctx, force_rerun=True, block=True
                )
            shutdown_verify_worker(wait=True)

            self.assertGreaterEqual(skipped, 1)
            for stage in ("tess_ffi_download", "mapping", "ps1_process"):
                self.assertEqual(
                    state.get_stage_run(run_id, label, stage).status,
                    STATUS_SKIPPED,
                )
            ps1_dl = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(ps1_dl.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "ps1_download"),
                SKIP_REASON_SUPERSEDED,
            )
            for stage in ("wcs_grouping", "downsample"):
                self.assertEqual(
                    state.get_stage_run(run_id, label, stage).status,
                    STATUS_PENDING,
                )
            self.assertTrue(state.deps_satisfied(run_id, label, "wcs_grouping"))
            self.assertFalse(state.deps_satisfied(run_id, label, "downsample"))

    def test_force_rerun_promotes_active_stages_when_prereqs_skipped(self):
        target = Target(20, 3, 3, 210.219333, 81.846589, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path,
                [target],
                active_stages=["wcs_grouping", "downsample"],
                force_rerun=True,
            )
            label = target.label()
            for stage in ("tess_ffi_download", "mapping", "ps1_download", "ps1_process"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)

            promoted = state.promote_stages(run_id)
            self.assertGreaterEqual(promoted, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "wcs_grouping").status,
                STATUS_READY,
            )
            self.assertEqual(
                state.get_stage_run(run_id, label, "downsample").status,
                STATUS_PENDING,
            )

            state.update_stage_status(run_id, label, "wcs_grouping", STATUS_SUCCESS, exit_code=0)
            state.promote_stages(run_id)
            self.assertEqual(
                state.get_stage_run(run_id, label, "downsample").status,
                STATUS_READY,
            )


def _minimal_run_setup(
    tmp_path: Path,
    targets: list[Target],
    *,
    active_stages: list[str],
    run_id: str = "run_a",
    force_rerun: bool = False,
):
    """Create run directory, state DB, and RunContext for scheduler tests."""
    runs_root = tmp_path / "runs"
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
    header = "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
    rows = [
        f"{t.sector},{t.camera},{t.ccd},{t.target_ra},{t.target_dec},{t.target_name},true"
        for t in targets
    ]
    (run_dir / "targets.csv").write_text(header + "\n".join(rows), encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    state = PipelineState(str(tmp_path / "state.sqlite"))
    state.create_run(
        run_id,
        str(cfg_path),
        str(run_dir / "targets.csv"),
        str(runs_root),
        targets,
        active_stages,
        force_rerun=force_rerun,
    )
    from syndiff_pipeline.template_creation.orchestration.run_context import resolve_run_context

    ctx = resolve_run_context(run_dir=run_dir)
    return state, ctx, run_id, runs_root


def _write_mapping_csv_and_manifest(
    tmp_path: Path,
    target: Target,
    manifest_path: Path,
    *,
    runs_root: Path | None = None,
) -> None:
    """Write mapping CSV plus a valid manifest at *manifest_path*."""
    from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config, resolve_config

    if runs_root is not None:
        cfg_path = runs_root / "run_a" / "config.yaml"
        cfg = load_runner_config(cfg_path)
    else:
        cfg = load_runner_config(tmp_path / "config.yaml")
    resolved = resolve_config(target, cfg)
    csv_path = (
        Path(resolved.mapping_root)
        / f"sector_{target.sector:04d}"
        / f"camera_{target.camera}"
        / f"ccd_{target.ccd}"
        / f"tess_s{target.sector:04d}_{target.camera}_{target.ccd}_master_skycells_list.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("NAME,projection\nskycell.0001.0001,0001\n", encoding="utf-8")
    write_manifest(manifest_path, resolved, "mapping", [str(csv_path)], 1, 1)


def _write_mapping_stable_manifest(tmp_path: Path, target: Target, runs_root: Path) -> None:
    """Write a valid stable mapping manifest plus on-disk CSV for *target*."""
    stable_path = logs.stable_stage_manifest_path(
        str(runs_root), target.label(), "mapping"
    )
    _write_mapping_csv_and_manifest(
        tmp_path, target, stable_path, runs_root=runs_root
    )


class TestSkipBeforePromote(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_promotion_blocked_until_skip_checked(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            stages = [
                "tess_ffi_download",
                "wcs_grouping",
                "mapping",
                "ps1_download",
                "ps1_process",
                "downsample",
            ]
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, [target], stages)
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status("run_a", label, stage, STATUS_SKIPPED, exit_code=0)

            promoted = state.promote_stages("run_a")

            self.assertEqual(promoted, 0)
            self.assertEqual(
                state.get_stage_run("run_a", label, "mapping").status, STATUS_PENDING
            )

    def test_pending_skip_then_promote_same_tick(self):
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

            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
                return_value=False,
            ):
                _resolve_external_and_pending_skips(state, run_id, ctx, force_rerun=False)

            self.assertTrue(state.external_verify_attempted(run_id, label, "mapping"))
            self.assertFalse(state.external_verify_complete(run_id, label, "mapping"))
            promoted = state.promote_stages(run_id)
            self.assertEqual(promoted, 0)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_PENDING
            )

    def test_mapping_skipped_before_launch_multi_target(self):
        targets = [
            Target(22, 3, 3, 228.0, 52.0, "2020dgc"),
            Target(23, 1, 3, 185.0, 5.3, "2020ftl"),
            Target(40, 1, 1, 292.6, 35.7, "2021udg"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path,
                targets,
                active_stages=["tess_ffi_download", "wcs_grouping", "mapping"],
            )
            for target in targets:
                label = target.label()
                _write_mapping_stable_manifest(tmp_path, target, runs_root)
                for stage in ("tess_ffi_download", "wcs_grouping"):
                    state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                    state.cache_external_check(run_id, label, stage, complete=True)

            ctx.cfg.verify_budget_per_tick = 2
            launch_mock = unittest.mock.Mock()
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler.launcher.launch_stage",
                launch_mock,
            ):
                for _ in range(5):
                    _tick_run(state, run_id, ctx)
                    if all(
                        state.get_stage_run(run_id, t.label(), "mapping").status
                        == STATUS_SKIPPED
                        for t in targets
                    ):
                        break

            launch_mock.assert_not_called()
            for target in targets:
                row = state.get_stage_run(run_id, target.label(), "mapping")
                self.assertEqual(row.status, STATUS_SKIPPED)


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
            from syndiff_pipeline.template_creation.orchestration.run_context import resolve_run_context

            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler._schedule_external_and_pending_skips",
                return_value=None,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler._verify_backlog",
                return_value=(0, 0),
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ):
                _tick_run(state, run_id, ctx)
            run = state.get_run(run_id)
            self.assertEqual(run["status"], "stalled")
            self.assertIn("waiting on", run["stall_reason"])


class TestAsyncVerify(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_tick_does_not_block_on_slow_verify(self):
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
                time.sleep(1.5)
                return False

            reconcile_calls: list[int] = []

            def track_reconcile(*_args, **_kwargs):
                reconcile_calls.append(1)
                return {}

            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler.reconcile_running_stages",
                side_effect=track_reconcile,
            ):
                started = time.monotonic()
                _tick_run(state, run_id, ctx)
                elapsed = time.monotonic() - started

            self.assertEqual(reconcile_calls, [1])
            self.assertLess(elapsed, 1.0)
            from syndiff_pipeline.template_creation.orchestration.verify_worker import get_verify_worker

            self.assertGreaterEqual(get_verify_worker().in_flight_count(run_id), 1)


class TestVerifyApplyGuards(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_stale_apply_rejected_after_retry(self):
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
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=False
                )
                state.reset_stage_for_retry(run_id, label, "mapping", reset_downstream=False)
                _cancel_verify_for_retry(
                    run_id, label, "mapping", reset_downstream=False
                )
                get_verify_worker().drain(
                    lambda outcome: _apply_verify_outcome(state, outcome),
                    run_id=run_id,
                    block=True,
                    block_timeout_s=3.0,
                )

            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.status, STATUS_PENDING)
            self.assertFalse(state.external_checked(run_id, label, "mapping"))

    def test_force_rerun_cancels_in_flight_verify(self):
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
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
                side_effect=slow_complete,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=False
                )
                state.apply_force_rerun(run_id, [label], ["mapping"])
                get_verify_worker().cancel_run(run_id)
                get_verify_worker().drain(
                    lambda outcome: _apply_verify_outcome(state, outcome),
                    run_id=run_id,
                    block=True,
                    block_timeout_s=3.0,
                )

            row = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(row.status, STATUS_PENDING)


class TestPerRunManifestBackfill(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_fast_path_backfills_stable_manifest_from_per_run(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            run_manifest = logs.stage_manifest_path(
                str(runs_root), run_id, label, "mapping"
            )
            _write_mapping_csv_and_manifest(
                tmp_path, target, run_manifest, runs_root=runs_root
            )
            stable_path = logs.stable_stage_manifest_path(
                str(runs_root), label, "mapping"
            )
            self.assertFalse(stable_path.is_file())

            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)

            _resolve_external_and_pending_skips(
                state, run_id, ctx, force_rerun=False, block=True
            )
            shutdown_verify_worker(wait=True)

            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_SKIPPED
            )
            self.assertTrue(stable_path.is_file())


class TestManifestFastPath(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_stable_manifest_skips_without_thread_pool(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            _write_mapping_stable_manifest(tmp_path, target, runs_root)
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)
            for row in state.list_stage_runs(run_id):
                if row.target_label == label and row.stage != "mapping":
                    state.cache_external_check(
                        run_id, label, row.stage, complete=False
                    )

            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
            ) as complete_mock:
                _resolve_external_and_pending_skips(
                    state, run_id, ctx, force_rerun=False
                )
                mapping_calls = [
                    c
                    for c in complete_mock.call_args_list
                    if len(c.args) >= 2 and c.args[1] == "mapping"
                ]
                self.assertEqual(mapping_calls, [])

            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_SKIPPED
            )


class TestApplyNoMainThreadCollect(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_apply_does_not_collect_artifacts_on_main_thread(self):
        import threading

        from syndiff_pipeline.template_creation.orchestration import verify as verify_mod

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

            main_tid = threading.get_ident()
            real_collect = verify_mod.collect_stage_artifacts

            def guard_collect(*args, **kwargs):
                if threading.get_ident() == main_tid:
                    raise AssertionError("collect_stage_artifacts on main thread")
                return real_collect(*args, **kwargs)

            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify_worker.stage_complete",
                return_value=True,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.collect_stage_artifacts",
                side_effect=guard_collect,
            ):
                _resolve_external_and_pending_skips(
                    state, run_id, ctx, force_rerun=False, block=True
                )

            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_SKIPPED
            )


class TestDaemonFlock(unittest.TestCase):
    def test_second_nonblocking_lock_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            with daemon.daemon_lock(handoff, blocking=True) as fd1:
                self.assertIsNotNone(fd1)
                with daemon.daemon_lock(handoff, blocking=False) as fd2:
                    self.assertIsNone(fd2)


class TestStopDaemon(unittest.TestCase):
    def test_cleans_stale_pid_file_when_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            pid_path = logs.daemon_pid_path(handoff)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("424242", encoding="utf-8")
            result = stop_daemon(handoff)
            self.assertFalse(result.was_running)
            self.assertTrue(result.stopped)
            self.assertFalse(result.force_killed)
            self.assertFalse(pid_path.is_file())

    def test_waits_for_graceful_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            pid_path = logs.daemon_pid_path(handoff)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("55555", encoding="utf-8")
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.is_process_alive",
                side_effect=[True, False],
            ) as alive, unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.terminate_process_tree",
            ) as term, unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.wait_for_process_exit",
                return_value=True,
            ) as wait:
                result = stop_daemon(handoff, term_timeout_s=0.1, kill_wait_s=0.1)
            self.assertTrue(result.was_running)
            self.assertTrue(result.stopped)
            self.assertFalse(result.force_killed)
            term.assert_called_once_with(55555, unittest.mock.ANY)
            wait.assert_called_once_with(55555, timeout_s=0.1)
            alive.assert_called()
            self.assertFalse(pid_path.is_file())

    def test_escalates_to_sigkill_when_term_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            pid_path = logs.daemon_pid_path(handoff)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("66666", encoding="utf-8")
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.is_process_alive",
                side_effect=[True, False],
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.terminate_process_tree",
            ) as term, unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.wait_for_process_exit",
                side_effect=[False, True],
            ) as wait:
                result = stop_daemon(handoff, term_timeout_s=0.1, kill_wait_s=0.1)
            self.assertTrue(result.force_killed)
            self.assertTrue(result.stopped)
            self.assertEqual(term.call_count, 2)
            term.assert_any_call(66666, unittest.mock.ANY)
            self.assertEqual(wait.call_count, 2)
            self.assertFalse(pid_path.is_file())

    def test_reports_failure_when_process_survives_sigkill(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            pid_path = logs.daemon_pid_path(handoff)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("77777", encoding="utf-8")
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.is_process_alive",
                return_value=True,
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.terminate_process_tree",
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.wait_for_process_exit",
                return_value=False,
            ):
                result = stop_daemon(handoff, term_timeout_s=0.1, kill_wait_s=0.1)
            self.assertTrue(result.was_running)
            self.assertFalse(result.stopped)
            self.assertTrue(result.force_killed)
            self.assertTrue(pid_path.is_file())

    def test_clears_liveness_after_sigkill_stop(self):
        """After stop (including SIGKILL), alive must be false immediately."""
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            pid_path = logs.daemon_pid_path(handoff)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("88888", encoding="utf-8")
            state = PipelineState(str(Path(handoff) / "pipeline_state.sqlite"))
            state.update_supervisor_heartbeat(88888)
            _write_local_heartbeat(handoff)
            self.addCleanup(
                lambda: logs.daemon_heartbeat_file(handoff).unlink(missing_ok=True)
            )
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.is_process_alive",
                side_effect=[True, False],
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.terminate_process_tree",
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler_control.daemon.wait_for_process_exit",
                side_effect=[False, True],
            ):
                result = stop_daemon(handoff, term_timeout_s=0.1, kill_wait_s=0.1)
            self.assertTrue(result.stopped)
            self.assertTrue(result.force_killed)
            self.assertFalse(logs.daemon_heartbeat_file(handoff).is_file())
            self.assertIsNone(state.get_supervisor_status())
            self.assertFalse(daemon_is_alive(handoff))

    def test_clears_stale_liveness_when_pid_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = tmp
            pid_path = logs.daemon_pid_path(handoff)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text("424242", encoding="utf-8")
            state = PipelineState(str(Path(handoff) / "pipeline_state.sqlite"))
            state.update_supervisor_heartbeat(424242)
            _write_local_heartbeat(handoff)
            self.addCleanup(
                lambda: logs.daemon_heartbeat_file(handoff).unlink(missing_ok=True)
            )
            result = stop_daemon(handoff)
            self.assertFalse(result.was_running)
            self.assertFalse(daemon_is_alive(handoff))
            self.assertIsNone(state.get_supervisor_status())


class TestRunFinalStatus(unittest.TestCase):
    def test_derive_run_final_status_canceled_beats_success(self):
        counts = {STATUS_SKIPPED: 1, STATUS_CANCELED: 5}
        self.assertEqual(derive_run_final_status(counts), RUN_CANCELED)

    def test_derive_run_final_status_failed_without_cancel(self):
        from syndiff_pipeline.template_creation.orchestration.state import RUN_FAILED

        counts = {STATUS_SUCCESS: 2, STATUS_FAILED: 1}
        self.assertEqual(derive_run_final_status(counts), RUN_FAILED)

    def test_derive_run_final_status_all_success_or_skipped(self):
        counts = {STATUS_SUCCESS: 3, STATUS_SKIPPED: 2}
        self.assertEqual(derive_run_final_status(counts), RUN_SUCCESS)

    def test_tick_run_does_not_mark_canceled_run_success(self):
        target = Target(20, 3, 3, 221.33, 38.73, "2020ghq")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path,
                [target],
                active_stages=["tess_ffi_download", "wcs_grouping", "mapping"],
            )
            label = target.label()
            state.update_stage_status(
                run_id, label, "tess_ffi_download", STATUS_SKIPPED, exit_code=0
            )
            state.update_stage_status(run_id, label, "wcs_grouping", STATUS_RUNNING)
            state.apply_cancel_run(run_id)
            self.assertEqual((state.get_run(run_id) or {}).get("status"), RUN_CANCELED)
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ):
                _tick_run(state, run_id, ctx)
            self.assertEqual((state.get_run(run_id) or {}).get("status"), RUN_CANCELED)


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


class TestNotSelectedSkips(unittest.TestCase):
    def test_upstream_stages_for_wcs_only(self):
        self.assertEqual(upstream_stages_for(["wcs_grouping"]), frozenset({"tess_ffi_download"}))

    def test_wcs_only_partial_ps1_zarr_does_not_stall(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["wcs_grouping"]
            )
            label = target.label()
            state.update_stage_status(run_id, label, "tess_ffi_download", STATUS_SKIPPED, exit_code=0)
            state.cache_external_check(run_id, label, "tess_ffi_download", complete=True)
            state.update_stage_status(run_id, label, "wcs_grouping", STATUS_SUCCESS, exit_code=0)
            state.cache_external_check(run_id, label, "ps1_download", complete=False)

            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.template_creation.orchestration.scheduler.launcher.launch_stage",
            ):
                _tick_run(state, run_id, ctx)

            ps1 = state.get_stage_run(run_id, label, "ps1_download")
            self.assertEqual(ps1.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "ps1_download"),
                SKIP_REASON_NOT_SELECTED,
            )
            self.assertEqual(state.get_run(run_id)["status"], RUN_SUCCESS)

    def test_ps1_process_only_keeps_ps1_download_external(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_process"]
            )
            label = target.label()
            state.cache_external_check(run_id, label, "ps1_download", complete=False)
            count = state.apply_not_selected_skips(run_id, ctx.targets, ctx.cfg)
            ps1 = state.get_stage_run(run_id, label, "ps1_download")
            down = state.get_stage_run(run_id, label, "downsample")
            self.assertEqual(ps1.status, STATUS_EXTERNAL)
            self.assertEqual(down.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "downsample"),
                SKIP_REASON_NOT_SELECTED,
            )
            self.assertGreater(count, 0)

    def test_status_grid_shows_na_for_not_selected(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["wcs_grouping"]
            )
            label = target.label()
            state.apply_not_selected_skips(run_id, ctx.targets, ctx.cfg)
            lines = format_status_grid(state, run_id)
            self.assertEqual(len(lines), 1)
            self.assertIn("map:n/a", lines[0])
            self.assertIn("ps1_dl:n/a", lines[0])


class TestSupersededSkips(unittest.TestCase):
    def test_ps1_download_superseded_when_ps1_process_skipped(self):
        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping", "downsample"]
            )
            label = target.label()
            state.update_stage_status(run_id, label, "wcs_grouping", STATUS_SKIPPED, exit_code=0)
            state.update_stage_status(run_id, label, "ps1_process", STATUS_SKIPPED, exit_code=0)
            state.cache_external_check(run_id, label, "ps1_download", complete=False)

            count = state.apply_superseded_skips(run_id, ctx.targets)
            ps1_dl = state.get_stage_run(run_id, label, "ps1_download")
            self.assertGreater(count, 0)
            self.assertEqual(ps1_dl.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "ps1_download"),
                SKIP_REASON_SUPERSEDED,
            )

            from syndiff_pipeline.template_creation.orchestration.state import artifact_verify_needed

            self.assertFalse(
                artifact_verify_needed(state, run_id, label, "ps1_download", ["mapping", "downsample"])
            )

    def test_tess_superseded_when_wcs_skipped(self):
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            label = target.label()
            state.update_stage_status(run_id, label, "wcs_grouping", STATUS_SKIPPED, exit_code=0)
            state.apply_superseded_skips(run_id, ctx.targets)
            tess = state.get_stage_run(run_id, label, "tess_ffi_download")
            self.assertEqual(tess.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "tess_ffi_download"),
                SKIP_REASON_SUPERSEDED,
            )

    def test_ps1_download_still_verified_without_satisfied_ps1_process(self):
        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_process"]
            )
            label = target.label()
            state.update_stage_status(run_id, label, "mapping", STATUS_SKIPPED, exit_code=0)
            state.update_stage_status(run_id, label, "wcs_grouping", STATUS_SKIPPED, exit_code=0)

            from syndiff_pipeline.template_creation.orchestration.state import artifact_verify_needed

            self.assertTrue(
                artifact_verify_needed(state, run_id, label, "ps1_download", ["ps1_process"])
            )

    def test_wcs_not_superseded_when_distant_ps1_process_skipped(self):
        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping", "downsample"]
            )
            label = target.label()
            state.update_stage_status(run_id, label, "ps1_process", STATUS_SKIPPED, exit_code=0)

            from syndiff_pipeline.template_creation.orchestration.state import artifact_verify_needed

            self.assertTrue(
                artifact_verify_needed(
                    state, run_id, label, "wcs_grouping", ["mapping", "downsample"]
                )
            )
            wcs = state.get_stage_run(run_id, label, "wcs_grouping")
            self.assertEqual(wcs.status, STATUS_EXTERNAL)


class TestPartialRunRetry(unittest.TestCase):
    def tearDown(self):
        reset_verify_worker_for_tests()

    def test_cancel_retry_partial_run_enqueues_ps1_verify(self):
        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping", "downsample"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
            state.update_stage_status(run_id, label, "mapping", STATUS_SUCCESS, exit_code=0)
            for stage in ("ps1_download", "ps1_process"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_skip_reason(run_id, label, stage, SKIP_REASON_ARTIFACTS)
                state.cache_external_check(run_id, label, stage, complete=True)
            state.update_stage_status(run_id, label, "mapping", STATUS_CANCELED, exit_code=143)
            state.update_stage_status(run_id, label, "downsample", STATUS_CANCELED, exit_code=143)

            state.reopen_failed_canceled(run_id)
            state.update_stage_status(run_id, label, "mapping", STATUS_SUCCESS, exit_code=0)

            candidates = _iter_verify_candidates(state, run_id, ctx, force_rerun=False)
            keys = {(c[0].target_label, c[0].stage) for c in candidates}
            self.assertIn((label, "ps1_download"), keys)

    def test_orphan_pending_shows_sc_q_in_status_grid(self):
        from syndiff_pipeline.template_creation.orchestration.run_report import format_status_grid

        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping", "downsample"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping", "mapping"):
                state.update_stage_status(run_id, label, stage, STATUS_SUCCESS, exit_code=0)
            state.update_stage_status(run_id, label, "ps1_download", STATUS_PENDING)
            state.update_stage_status(run_id, label, "ps1_process", STATUS_PENDING)

            lines = format_status_grid(state, run_id)
            self.assertIn("ps1_dl:sc_q", lines[0])
            self.assertIn("ps1_pr:sc_q", lines[0])


class TestVerifyDisplay(unittest.TestCase):
    def test_status_grid_shows_sc_q_for_external_scan_queue(self):
        from syndiff_pipeline.template_creation.orchestration.run_report import format_status_grid

        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["ps1_process"]
            )
            label = target.label()
            for stage in ("tess_ffi_download", "wcs_grouping", "mapping"):
                state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
                state.cache_external_check(run_id, label, stage, complete=True)
            lines = format_status_grid(state, run_id)
            self.assertIn("ps1_dl:sc_q", lines[0])

    def test_promote_stages_requires_complete_verify(self):
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
            state.cache_external_check(run_id, label, "mapping", complete=False)
            promoted = state.promote_stages(run_id)
            self.assertEqual(promoted, 0)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_PENDING
            )


if __name__ == "__main__":
    unittest.main()
