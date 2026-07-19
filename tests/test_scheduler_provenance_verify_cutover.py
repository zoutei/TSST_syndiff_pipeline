"""Tests for the ps1_process checkpoint-first verify cutover (PR3, plan §11/§19).

``scheduler._run_verify_pass`` gains one new branch: for the ``ps1_process``
stage specifically, before the legacy ``check_manifests_only`` /
``stage_absence_probe`` / background ``VerifyTask`` path, it recomputes the
``scc_assembly`` fingerprint fresh from the current resolved config and
checks it against the real ``ProvenanceStore``. A hit routes through the
existing ``_apply_verify_outcome`` with a synthesized "ok" outcome -- no
directory walk. A miss (or any failure) falls open, unchanged, to the legacy
path. Every other stage's verify flow must be byte-for-byte untouched.

These tests run against the real ``provenance.store``/``provenance.ingest``/
``provenance.publish``/``provenance_checkpoint``/``scc_paths`` modules (all
landed as of PR1/PR2 on this branch) -- true end-to-end coverage of the
checkpoint-first branch, not a mocked contract.
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
    STATUS_READY,
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
from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    emit_scc_assembly_checkpoint,
    expected_scc_assembly_fingerprint,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config
from tests.test_daemon_behavior import _minimal_run_setup

_UPSTREAM_STAGES = ("tess_ffi_download", "mapping", "ps1_download", "remap")


def _publish_and_ingest_checkpoint(resolved) -> str:
    """Emit the ``scc_assembly`` checkpoint for *resolved* and drain it into
    the real provenance DB, exactly as the supervisor's spool drain would
    (§10/§15). Returns the fingerprint that was published."""
    emit_scc_assembly_checkpoint(resolved)
    store = ProvenanceStore(str(provenance_db_path(resolved.data_root)))
    drain_spool(store, provenance_spool_dir(resolved.data_root))
    return expected_scc_assembly_fingerprint(resolved)


def _setup_run_with_ps1_process_external(tmp_path: Path, target: Target):
    """Build a run where ``ps1_process`` is the sole EXTERNAL verify
    candidate: ``downsample`` is the active (pending) stage, and every other
    upstream stage is pre-marked skipped/verified so it never enters
    ``_iter_verify_candidates``."""
    state, ctx, run_id, _runs_root = _minimal_run_setup(
        tmp_path, [target], active_stages=["downsample"]
    )
    label = target.label()
    for stage in _UPSTREAM_STAGES:
        state.update_stage_status(run_id, label, stage, STATUS_SKIPPED, exit_code=0)
        state.cache_external_check(run_id, label, stage, complete=True)
    ps1 = state.get_stage_run(run_id, label, "ps1_process")
    assert ps1.status == STATUS_EXTERNAL, ps1.status
    return state, ctx, run_id, label


class _BaseVerifyCutoverTest(unittest.TestCase):
    def setUp(self):
        reset_verify_worker_for_tests()

    def tearDown(self):
        shutdown_verify_worker(wait=False)
        reset_verify_worker_for_tests()


class TestCheckpointHitShortCircuits(_BaseVerifyCutoverTest):
    """(a) A checkpoint hit skips the legacy scan entirely."""

    def test_hit_skips_ps1_process_without_legacy_scan_calls(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            resolved = resolve_config(target, ctx.cfg)
            _publish_and_ingest_checkpoint(resolved)

            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.check_manifests_only"
            ) as mock_manifests, mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.stage_absence_probe"
            ) as mock_probe, mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.ArtifactVerifyWorker.schedule"
            ) as mock_schedule:
                skipped = _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertGreaterEqual(skipped, 1)
            mock_manifests.assert_not_called()
            mock_probe.assert_not_called()
            mock_schedule.assert_not_called()

            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertEqual(ps1.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "ps1_process"),
                SKIP_REASON_ARTIFACTS,
            )
            self.assertTrue(
                state.external_verify_complete(run_id, label, "ps1_process")
            )

    def test_hit_causes_zero_filesystem_scan_calls(self):
        """Fault-injection: the hit path must never walk the convolved.zarr
        tree. Raise from any of the legacy scan primitives; the hit path
        must never reach them."""
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            resolved = resolve_config(target, ctx.cfg)
            _publish_and_ingest_checkpoint(resolved)

            def _boom(*_a, **_kw):
                raise AssertionError(
                    "directory walk invoked on a checkpoint-hit verify pass"
                )

            with mock.patch.object(
                os, "scandir", side_effect=_boom
            ), mock.patch.object(
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
            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertEqual(ps1.status, STATUS_SKIPPED)


class TestCheckpointMissFallsOpen(_BaseVerifyCutoverTest):
    """(b) A checkpoint miss (no matching fp in the store) falls through to
    the exact legacy path, unchanged."""

    def test_miss_reaches_legacy_check_manifests_only(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            # No checkpoint emitted at all -- store/db do not exist yet.

            real_check_manifests_only = None
            from syndiff_pipeline.template_creation.orchestration import verify as verify_mod

            real_check_manifests_only = verify_mod.check_manifests_only
            calls: list[str] = []

            def spy(resolved, stage, **kwargs):
                calls.append(stage)
                return real_check_manifests_only(resolved, stage, **kwargs)

            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.check_manifests_only",
                side_effect=spy,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertIn("ps1_process", calls)
            # Not a checkpoint skip: fell through to the real legacy path.
            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertNotEqual(ps1.status, STATUS_SKIPPED)


class TestConfigDriftForcesMiss(_BaseVerifyCutoverTest):
    """(c) Config drift between the emitted checkpoint and the current
    resolved config yields a different expected fingerprint -> miss."""

    def test_psf_sigma_drift_produces_a_miss_and_falls_open(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            # Publish a checkpoint under the *old* config (default psf_sigma).
            old_resolved = resolve_config(target, ctx.cfg)
            old_fp = _publish_and_ingest_checkpoint(old_resolved)

            # Drift the live config's psf_sigma -- the fingerprint recomputed
            # from *current* config must differ, and therefore miss.
            ctx.cfg.stages.ps1_process.psf_sigma = (
                ctx.cfg.stages.ps1_process.psf_sigma + 37.0
            )
            new_resolved = resolve_config(target, ctx.cfg)
            new_fp = expected_scc_assembly_fingerprint(new_resolved)
            self.assertNotEqual(old_fp, new_fp)

            from syndiff_pipeline.template_creation.orchestration import verify as verify_mod

            real_check_manifests_only = verify_mod.check_manifests_only
            calls: list[str] = []

            def spy(resolved, stage, **kwargs):
                calls.append(stage)
                return real_check_manifests_only(resolved, stage, **kwargs)

            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.check_manifests_only",
                side_effect=spy,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertIn("ps1_process", calls)
            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertNotEqual(ps1.status, STATUS_SKIPPED)


class TestProvenanceUnavailableFallsOpen(_BaseVerifyCutoverTest):
    """(d) Provenance package import failure / missing DB falls open cleanly
    -- the scheduler must never crash."""

    def test_cold_scc_with_no_provenance_db_falls_open(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            resolved = resolve_config(target, ctx.cfg)
            self.assertFalse(provenance_db_path(resolved.data_root).exists())

            try:
                skipped = _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )
            except Exception as exc:  # pragma: no cover - the assertion is the point
                self.fail(f"_run_verify_pass raised on a cold SCC: {exc!r}")
            self.assertIsInstance(skipped, int)

            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertNotEqual(ps1.status, STATUS_SKIPPED)

    def test_provenance_package_import_failure_falls_open(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            resolved = resolve_config(target, ctx.cfg)
            _publish_and_ingest_checkpoint(resolved)

            # Simulate the provenance store module being unimportable (mid
            # authoring window / broken install): setting it to None in
            # sys.modules makes `import` raise ImportError inside the
            # checkpoint-hit helper, which must swallow it and fall open.
            with mock.patch.dict(
                sys.modules,
                {"syndiff_pipeline.common.provenance.store": None},
            ):
                try:
                    _run_verify_pass(
                        state, run_id, ctx, force_rerun=False, budget=8, block=True
                    )
                except Exception as exc:  # pragma: no cover
                    self.fail(
                        f"_run_verify_pass raised when provenance.store was"
                        f" unimportable: {exc!r}"
                    )

            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            # Fell open to the legacy path -- not a checkpoint short-circuit.
            self.assertNotEqual(ps1.status, STATUS_SKIPPED)


class TestCheckpointHitPromotesLikeLegacyScan(_BaseVerifyCutoverTest):
    """(e) A checkpoint hit drives promote_stages to the same state a
    legacy-scan success would (integration-style)."""

    def test_hit_promotes_downsample_to_ready(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            resolved = resolve_config(target, ctx.cfg)
            _publish_and_ingest_checkpoint(resolved)

            skipped = _run_verify_pass(
                state, run_id, ctx, force_rerun=False, budget=8, block=True
            )
            self.assertGreaterEqual(skipped, 1)
            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertEqual(ps1.status, STATUS_SKIPPED)

            target_stages = {label: resolve_config(target, ctx.cfg).stages}
            promoted = state.promote_stages(run_id, target_stages)
            self.assertGreaterEqual(promoted, 1)
            downsample = state.get_stage_run(run_id, label, "downsample")
            self.assertEqual(downsample.status, STATUS_READY)


class TestOtherStagesUntouched(_BaseVerifyCutoverTest):
    """(f) The checkpoint-first branch is stage-gated to ps1_process only;
    every other stage's verify path takes zero new code."""

    def test_checkpoint_helper_is_only_invoked_for_ps1_process(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # active_stages=["downsample"] puts tess_ffi_download, mapping,
            # ps1_download, ps1_process, remap all in the verify closure as
            # candidates (none pre-skipped here) -- a broad sweep across
            # every non-ps1_process stage.
            state, ctx, run_id, _runs_root = _minimal_run_setup(
                tmp_path, [target], active_stages=["downsample"]
            )
            calls: list[str] = []

            def spy(key, resolved, stable_path):
                calls.append(key.stage)
                return None  # always miss -- let the legacy path continue

            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler._ps1_process_checkpoint_hit",
                side_effect=spy,
            ):
                _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=16, block=True
                )

            self.assertIn("ps1_process", calls)
            for stage in ("mapping", "ps1_download", "remap", "tess_ffi_download", "downsample"):
                self.assertNotIn(
                    stage,
                    calls,
                    msg=f"checkpoint-first branch must not run for stage={stage!r}",
                )

    def test_stage_constant_matches_ps1_process_only(self):
        from syndiff_pipeline.common.orchestration import scheduler

        self.assertEqual(scheduler._CHECKPOINT_FIRST_STAGE, "ps1_process")


if __name__ == "__main__":
    unittest.main()
