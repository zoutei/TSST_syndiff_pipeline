"""Fault-injection tests for all template checkpoint-first verify paths (BK-3).

For each checkpoint-enabled template stage (``tess_ffi_download``,
``mapping``, ``remap``, ``downsample``, ``ps1_process``), a provenance hit
must short-circuit the legacy verify path: no ``check_manifests_only``,
``stage_absence_probe``, or background ``VerifyTask.schedule`` calls, and
``cache_external_check`` / stage marked skipped via ``_apply_verify_outcome``.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.scheduler import _run_verify_pass
from syndiff_pipeline.common.orchestration.state import (
    SKIP_REASON_ARTIFACTS,
    STATUS_EXTERNAL,
    STATUS_SKIPPED,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.orchestration.verify_worker import (
    reset_verify_worker_for_tests,
    shutdown_verify_worker,
)
from syndiff_pipeline.common.provenance.ingest import drain_spool
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path, provenance_spool_dir
from syndiff_pipeline.template_creation.orchestration import provenance_checkpoint
from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config
from tests.test_daemon_behavior import _minimal_run_setup

_CHECKPOINT_CASES = (
    ("tess_ffi_download", "emit_ffi_set_checkpoint", ["mapping"]),
    ("mapping", "emit_mapping_checkpoint", ["downsample"]),
    ("remap", "emit_remap_store_checkpoint", ["downsample"]),
    ("downsample", "emit_downsample_checkpoint", ["diff"]),
    ("ps1_process", "emit_scc_assembly_checkpoint", ["downsample"]),
)

_TEMPLATE_STAGES = (
    "tess_ffi_download",
    "mapping",
    "ps1_download",
    "ps1_process",
    "remap",
    "downsample",
)


def _publish_and_ingest(resolved, emit_name: str) -> str:
    emit_fn = getattr(provenance_checkpoint, emit_name)
    expected_fn_name = emit_name.replace("emit_", "expected_").replace(
        "_checkpoint", "_fingerprint"
    )
    expected_fn = getattr(provenance_checkpoint, expected_fn_name)
    emit_fn(resolved)
    store = ProvenanceStore(str(provenance_db_path(resolved.data_root)))
    drain_spool(store, provenance_spool_dir(resolved.data_root))
    return expected_fn(resolved)


_STAGE_EMITTERS = {
    "tess_ffi_download": "emit_ffi_set_checkpoint",
    "mapping": "emit_mapping_checkpoint",
    "remap": "emit_remap_store_checkpoint",
    "downsample": "emit_downsample_checkpoint",
    "ps1_process": "emit_scc_assembly_checkpoint",
}


def _ingest_checkpoints_for_external_stages(state, run_id, label, resolved) -> None:
    """Emit+ingest checkpoints for every EXTERNAL stage that has an emitter.

    Direct dependents of the stage under test must stay EXTERNAL (so
    ``artifact_verify_needed`` stays True). Emitting their checkpoints too
    keeps the verify pass on the indexed path without falling open to
    ``check_manifests_only``.
    """
    for stage, emit_name in _STAGE_EMITTERS.items():
        row = state.get_stage_run(run_id, label, stage)
        if row is None or row.status != STATUS_EXTERNAL:
            continue
        _publish_and_ingest(resolved, emit_name)


def _setup_run_with_external_stage(
    tmp_path: Path,
    target: Target,
    *,
    external_stage: str,
    active_stages: list[str],
):
    """Run where *external_stage* remains an EXTERNAL verify candidate.

    ``artifact_verify_needed`` returns False once any *direct dependent* in
    the verify closure is already success/skipped. So we must not mark active
    stages or direct dependents of *external_stage* as skipped — only mark
    unrelated / upstream stages.
    """
    state, ctx, run_id, _runs_root = _minimal_run_setup(
        tmp_path, [target], active_stages=active_stages
    )
    label = target.label()
    active = set(active_stages)
    dependents = set(state.pipeline_spec.direct_dependents(external_stage))
    for stage in _TEMPLATE_STAGES:
        if stage == external_stage or stage in active or stage in dependents:
            continue
        state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
        state.cache_external_check(run_id, label, stage, complete=True)
    row = state.get_stage_run(run_id, label, external_stage)
    assert row.status == STATUS_EXTERNAL, (external_stage, row.status)
    return state, ctx, run_id, label


class _BaseCheckpointTest(unittest.TestCase):
    def setUp(self):
        reset_verify_worker_for_tests()

    def tearDown(self):
        shutdown_verify_worker(wait=False)
        reset_verify_worker_for_tests()


def _parametrize_checkpoint_cases(cases):
    def decorator(cls):
        for stage, emit_name, active_stages in cases:
            test_name = f"test_hit_skips_legacy_verify__{stage}"

            def make_test(s=stage, e=emit_name, active=active_stages):
                def test_method(self):
                    target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp_path = Path(tmp)
                        state, ctx, run_id, label = _setup_run_with_external_stage(
                            tmp_path,
                            target,
                            external_stage=s,
                            active_stages=active,
                        )
                        resolved = resolve_config(target, ctx.cfg)
                        _ingest_checkpoints_for_external_stages(
                            state, run_id, label, resolved
                        )

                        with mock.patch(
                            "syndiff_pipeline.template_creation.orchestration.verify.check_manifests_only"
                        ) as mock_manifests, mock.patch(
                            "syndiff_pipeline.template_creation.orchestration.verify.stage_absence_probe"
                        ) as mock_probe, mock.patch(
                            "syndiff_pipeline.common.orchestration.verify_worker.ArtifactVerifyWorker.schedule"
                        ) as mock_schedule:
                            skipped = _run_verify_pass(
                                state,
                                run_id,
                                ctx,
                                force_rerun=False,
                                budget=8,
                                block=True,
                            )

                        self.assertGreaterEqual(skipped, 1, msg=s)
                        mock_manifests.assert_not_called()
                        mock_probe.assert_not_called()
                        mock_schedule.assert_not_called()

                        row = state.get_stage_run(run_id, label, s)
                        self.assertEqual(row.status, STATUS_SKIPPED, msg=s)
                        self.assertEqual(
                            state.get_skip_reason(run_id, label, s),
                            SKIP_REASON_ARTIFACTS,
                        )
                        self.assertTrue(
                            state.external_verify_complete(run_id, label, s)
                        )

                return test_method

            setattr(cls, test_name, make_test())
        return cls

    return decorator


@_parametrize_checkpoint_cases(_CHECKPOINT_CASES)
class TestAllTemplateCheckpointHits(_BaseCheckpointTest):
    pass


class TestCheckpointFilesystemScanGuard(_BaseCheckpointTest):
    def test_mapping_hit_no_filesystem_scan(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_external_stage(
                tmp_path,
                target,
                external_stage="mapping",
                active_stages=["downsample"],
            )
            resolved = resolve_config(target, ctx.cfg)
            _ingest_checkpoints_for_external_stages(state, run_id, label, resolved)

            def _boom(*_a, **_kw):
                raise AssertionError(
                    "directory walk invoked on a checkpoint-hit verify pass"
                )

            with mock.patch.object(os, "scandir", side_effect=_boom), mock.patch.object(
                Path, "iterdir", side_effect=_boom
            ):
                skipped = _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertGreaterEqual(skipped, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "mapping").status, STATUS_SKIPPED
            )

    def test_remap_hit_no_filesystem_scan(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_external_stage(
                tmp_path,
                target,
                external_stage="remap",
                active_stages=["downsample"],
            )
            resolved = resolve_config(target, ctx.cfg)
            _ingest_checkpoints_for_external_stages(state, run_id, label, resolved)

            def _boom(*_a, **_kw):
                raise AssertionError(
                    "directory walk invoked on a checkpoint-hit verify pass"
                )

            with mock.patch.object(os, "scandir", side_effect=_boom), mock.patch.object(
                Path, "iterdir", side_effect=_boom
            ):
                skipped = _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertGreaterEqual(skipped, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "remap").status, STATUS_SKIPPED
            )

    def test_downsample_hit_no_filesystem_scan(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_external_stage(
                tmp_path,
                target,
                external_stage="downsample",
                active_stages=["diff"],
            )
            resolved = resolve_config(target, ctx.cfg)
            _ingest_checkpoints_for_external_stages(state, run_id, label, resolved)

            def _boom(*_a, **_kw):
                raise AssertionError(
                    "directory walk invoked on a checkpoint-hit verify pass"
                )

            with mock.patch.object(os, "scandir", side_effect=_boom), mock.patch.object(
                Path, "iterdir", side_effect=_boom
            ):
                skipped = _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertGreaterEqual(skipped, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "downsample").status,
                STATUS_SKIPPED,
            )

    def test_ps1_process_hit_no_filesystem_scan(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_external_stage(
                tmp_path,
                target,
                external_stage="ps1_process",
                active_stages=["downsample"],
            )
            resolved = resolve_config(target, ctx.cfg)
            _ingest_checkpoints_for_external_stages(state, run_id, label, resolved)

            def _boom(*_a, **_kw):
                raise AssertionError(
                    "directory walk invoked on a checkpoint-hit verify pass"
                )

            with mock.patch.object(os, "scandir", side_effect=_boom), mock.patch.object(
                Path, "iterdir", side_effect=_boom
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify._count_convolved_data_arrays",
                side_effect=_boom,
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.expected_ps1_process_skycells",
                side_effect=_boom,
            ):
                skipped = _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertGreaterEqual(skipped, 1)
            self.assertEqual(
                state.get_stage_run(run_id, label, "ps1_process").status,
                STATUS_SKIPPED,
            )


class TestCheckpointStageGating(_BaseCheckpointTest):
    def test_checkpoint_helper_not_invoked_for_ps1_download(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["downsample"]
            )
            calls: list[str] = []

            def spy(stage, key, resolved, stable_path):
                calls.append(stage)
                return None

            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler._checkpoint_hit",
                side_effect=spy,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=True
                )

            for stage in (
                "tess_ffi_download",
                "mapping",
                "remap",
                "ps1_process",
            ):
                self.assertIn(stage, calls)
            # Active stages are not verified. ps1_download has no emitter;
            # _checkpoint_hit may still be entered and return None immediately.
            self.assertNotIn("downsample", calls)
            from syndiff_pipeline.common.orchestration import scheduler

            self.assertNotIn(
                "ps1_download", scheduler._CHECKPOINT_STAGE_FINGERPRINTS
            )


if __name__ == "__main__":
    unittest.main()
