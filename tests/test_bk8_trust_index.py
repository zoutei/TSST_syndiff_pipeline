"""BK-8: ``bookkeeping.trust_index`` gates manifest dual-write and legacy scans."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.scheduler import _run_verify_pass
from syndiff_pipeline.common.orchestration.state import STATUS_EXTERNAL, STATUS_SKIPPED
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.orchestration.verify_worker import (
    reset_verify_worker_for_tests,
    shutdown_verify_worker,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    RunnerConfig,
    _parse_bookkeeping_trust_index,
    resolve_config,
)
from syndiff_pipeline.template_creation.orchestration.verify import (
    collect_stage_artifacts,
    verify_ps1_process,
)
from tests.test_daemon_behavior import _minimal_run_setup
from tests.test_scheduler_provenance_verify_cutover import (
    _publish_and_ingest_checkpoint,
    _setup_run_with_ps1_process_external,
)


def _run_main(stage: str, *, trust_index: bool, emit_side_effect=None):
    fake_ctx = mock.Mock()
    fake_ctx.cfg = mock.Mock(
        bookkeeping_trust_index=trust_index,
        runs_dir=mock.Mock(return_value="/runs"),
    )
    fake_ctx.targets = [mock.Mock(label=mock.Mock(return_value="t1"))]
    fake_resolved = mock.Mock()

    with mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.resolve_run_context",
        return_value=fake_ctx,
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.resolve_config",
        return_value=fake_resolved,
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.dispatch.execute_stage",
        return_value=None,
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.logs.stage_log",
        return_value=mock.Mock(
            __enter__=mock.Mock(return_value=mock.Mock()),
            __exit__=mock.Mock(return_value=False),
        ),
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage._write_status",
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.collect_stage_artifacts",
        return_value=(1, 1, ["/artifact"]),
    ) as collect_mock, mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.write_manifest",
    ) as write_mock, mock.patch(
        "syndiff_pipeline.template_creation.orchestration.provenance_checkpoint.emit_scc_assembly_checkpoint",
        side_effect=emit_side_effect,
    ) as emit_mock:
        from syndiff_pipeline.common.orchestration import run_stage

        rc = run_stage.main(
            [
                "--run-id",
                "run_a",
                "--stage",
                stage,
                "--run-dir",
                "/runs/run_a",
                "--target-label",
                "t1",
                "--launch-token",
                "tok",
            ]
        )
    return rc, collect_mock, write_mock, emit_mock


class TestBookkeepingTrustIndexConfig(unittest.TestCase):
    def test_defaults_false(self):
        self.assertFalse(_parse_bookkeeping_trust_index({}))
        self.assertFalse(RunnerConfig().bookkeeping_trust_index)

    def test_nested_bookkeeping_key(self):
        self.assertTrue(
            _parse_bookkeeping_trust_index({"bookkeeping": {"trust_index": True}})
        )

    def test_top_level_alias(self):
        self.assertTrue(_parse_bookkeeping_trust_index({"bookkeeping_trust_index": True}))


class TestRunStageManifestDualWrite(unittest.TestCase):
    def test_flag_off_dual_writes_manifest_and_checkpoint(self):
        rc, collect_mock, write_mock, emit_mock = _run_main(
            "ps1_process", trust_index=False
        )
        self.assertEqual(rc, 0)
        collect_mock.assert_called_once()
        write_mock.assert_called()
        emit_mock.assert_called_once()

    def test_flag_on_skips_manifest_but_emits_checkpoint(self):
        rc, collect_mock, write_mock, emit_mock = _run_main(
            "ps1_process", trust_index=True
        )
        self.assertEqual(rc, 0)
        collect_mock.assert_not_called()
        write_mock.assert_not_called()
        emit_mock.assert_called_once()


class TestSchedulerTrustIndexOnly(unittest.TestCase):
    def setUp(self):
        reset_verify_worker_for_tests()

    def tearDown(self):
        shutdown_verify_worker(wait=False)
        reset_verify_worker_for_tests()

    def test_miss_does_not_fall_open_to_legacy_scan(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            ctx.cfg.bookkeeping_trust_index = True

            with self.assertLogs(
                "syndiff_pipeline.common.orchestration.scheduler", level="WARNING"
            ) as caplog:
                with mock.patch(
                    "syndiff_pipeline.template_creation.orchestration.verify.check_manifests_only"
                ) as mock_manifests, mock.patch(
                    "syndiff_pipeline.template_creation.orchestration.verify.stage_absence_probe"
                ) as mock_probe, mock.patch(
                    "syndiff_pipeline.common.orchestration.verify_worker.ArtifactVerifyWorker.schedule"
                ) as mock_schedule:
                    _run_verify_pass(
                        state, run_id, ctx, force_rerun=False, budget=8, block=True
                    )

            mock_manifests.assert_not_called()
            mock_probe.assert_not_called()
            mock_schedule.assert_not_called()
            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertNotEqual(ps1.status, STATUS_SKIPPED)
            warn_text = "\n".join(caplog.output)
            self.assertIn("Checkpoint index miss for stage ps1_process", warn_text)
            self.assertIn("syndiff bookkeeping reindex", warn_text)
            self.assertIn("bookkeeping.trust_index", warn_text)

    def test_store_unavailable_logs_distinct_warning(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            ctx.cfg.bookkeeping_trust_index = True

            with self.assertLogs(
                "syndiff_pipeline.common.orchestration.scheduler", level="WARNING"
            ) as caplog:
                with mock.patch(
                    "syndiff_pipeline.common.provenance.store.ProvenanceStore",
                    side_effect=OSError("provenance.db locked"),
                ), mock.patch(
                    "syndiff_pipeline.template_creation.orchestration.verify.check_manifests_only"
                ) as mock_manifests:
                    _run_verify_pass(
                        state, run_id, ctx, force_rerun=False, budget=8, block=True
                    )

            mock_manifests.assert_not_called()
            warn_text = "\n".join(caplog.output)
            self.assertIn("Provenance store unavailable for checkpoint verify", warn_text)
            self.assertIn("ps1_process", warn_text)
            self.assertNotIn("Checkpoint index miss", warn_text)
            ps1 = state.get_stage_run(run_id, label, "ps1_process")
            self.assertNotEqual(ps1.status, STATUS_SKIPPED)

    def test_hit_still_skips_legacy_scan(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, label = _setup_run_with_ps1_process_external(
                tmp_path, target
            )
            ctx.cfg.bookkeeping_trust_index = True
            resolved = resolve_config(target, ctx.cfg)
            _publish_and_ingest_checkpoint(resolved)

            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.check_manifests_only"
            ) as mock_manifests:
                skipped = _run_verify_pass(
                    state, run_id, ctx, force_rerun=False, budget=8, block=True
                )

            self.assertGreaterEqual(skipped, 1)
            mock_manifests.assert_not_called()
            self.assertEqual(
                state.get_stage_run(run_id, label, "ps1_process").status,
                STATUS_SKIPPED,
            )


class TestVerifyPs1ProcessTrustIndex(unittest.TestCase):
    def test_indexed_path_skips_convolved_scandir(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = RunnerConfig(
                data_root=str(tmp_path / "data"),
                workspace_root=str(tmp_path),
                bookkeeping_trust_index=True,
            )
            resolved = resolve_config(target, cfg)

            def _boom(*_a, **_kw):
                raise AssertionError("convolved scandir must not run with trust_index")

            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify._count_convolved_data_arrays",
                side_effect=_boom,
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.checkpoint_stage_indexed",
                return_value=True,
            ):
                result = verify_ps1_process(resolved, runner_cfg=cfg)

            self.assertTrue(result.ok)

    def test_collect_stage_artifacts_skips_scandir(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = RunnerConfig(
                data_root=str(tmp_path / "data"),
                workspace_root=str(tmp_path),
                bookkeeping_trust_index=True,
            )
            resolved = resolve_config(target, cfg)

            def _boom(*_a, **_kw):
                raise AssertionError("convolved scandir must not run with trust_index")

            with mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify._count_convolved_data_arrays",
                side_effect=_boom,
            ), mock.patch(
                "syndiff_pipeline.template_creation.orchestration.verify.checkpoint_stage_indexed",
                return_value=False,
            ):
                expected, produced, _artifacts = collect_stage_artifacts(
                    resolved, "ps1_process", runner_cfg=cfg
                )

            self.assertEqual((expected, produced), (1, 0))


if __name__ == "__main__":
    unittest.main()
