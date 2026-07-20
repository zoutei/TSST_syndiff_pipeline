"""Tests for scheduler.skip_artifact_verify (trust upstream without scanning)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.run_setup import apply_post_create_run_setup
from syndiff_pipeline.common.orchestration.scheduler import _tick_run
from syndiff_pipeline.common.orchestration.state import (
    SKIP_REASON_TRUSTED,
    STATUS_EXTERNAL,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SKIPPED,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.orchestration.verify_worker import (
    reset_verify_worker_for_tests,
    try_get_verify_worker,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    RunnerConfig,
    load_and_materialize_runner_config,
    runner_config_to_dict,
    write_runner_config,
)
from tests.test_daemon_behavior import _minimal_run_setup


class TestSkipArtifactVerifyConfig(unittest.TestCase):
    def test_roundtrip_scheduler_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            cfg = RunnerConfig(
                data_root=str(Path(tmp) / "data"),
                workspace_root=tmp,
                skip_artifact_verify=True,
            )
            write_runner_config(cfg, path)
            loaded = load_and_materialize_runner_config(path)
            self.assertTrue(loaded.skip_artifact_verify)
            serialized = runner_config_to_dict(loaded)
            self.assertTrue(serialized["scheduler"]["skip_artifact_verify"])


class TestTrustExternalArtifacts(unittest.TestCase):
    def test_trusts_external_upstream_for_mapping_run(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            apply_post_create_run_setup(
                state, run_id, ctx.targets, ctx.cfg, ["mapping"]
            )
            label = target.label()
            tess = state.get_stage_run(run_id, label, "tess_ffi_download")
            self.assertEqual(tess.status, STATUS_EXTERNAL)

            count = state.trust_external_artifacts(
                run_id, ctx.targets, ["mapping"]
            )
            self.assertEqual(count, 1)
            tess = state.get_stage_run(run_id, label, "tess_ffi_download")
            self.assertEqual(tess.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "tess_ffi_download"),
                SKIP_REASON_TRUSTED,
            )
            # Idempotent
            self.assertEqual(
                state.trust_external_artifacts(run_id, ctx.targets, ["mapping"]),
                0,
            )

            target_stages = {
                label: resolve_stages(ctx, target)
            }
            promoted = state.promote_stages(run_id, target_stages)
            self.assertGreaterEqual(promoted, 1)
            mapping = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(mapping.status, STATUS_READY)


def resolve_stages(ctx, target):
    from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

    return resolve_config(target, ctx.cfg).stages


class TestSkipArtifactVerifyTick(unittest.TestCase):
    def tearDown(self) -> None:
        reset_verify_worker_for_tests()

    def test_tick_trusts_and_promotes_when_enabled(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            apply_post_create_run_setup(
                state, run_id, ctx.targets, ctx.cfg, ["mapping"]
            )
            ctx.cfg.skip_artifact_verify = True
            label = target.label()

            launch_calls: list[str] = []

            def fake_launch(*_args, **kwargs):
                from syndiff_pipeline.common.orchestration.launcher import LaunchDescriptor

                launch_calls.append(kwargs["stage"])
                return LaunchDescriptor(
                    executor="local",
                    native_id=12345,
                    launch_token=kwargs["launch_token"],
                    submit_epoch=0.0,
                )

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=fake_launch,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler._schedule_external_and_pending_skips",
            ) as mock_verify:
                _tick_run(state, run_id, ctx)
                mock_verify.assert_not_called()

            tess = state.get_stage_run(run_id, label, "tess_ffi_download")
            self.assertEqual(tess.status, STATUS_SKIPPED)
            self.assertEqual(
                state.get_skip_reason(run_id, label, "tess_ffi_download"),
                SKIP_REASON_TRUSTED,
            )
            mapping = state.get_stage_run(run_id, label, "mapping")
            self.assertIn(mapping.status, (STATUS_READY, STATUS_RUNNING))
            self.assertEqual(launch_calls, ["mapping"])
            worker = try_get_verify_worker()
            if worker is not None:
                self.assertEqual(worker.in_flight_count(run_id), 0)

    def test_tick_still_scans_when_disabled(self):
        target = Target(20, 3, 3, 210.0, 81.0, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, _runs = _minimal_run_setup(
                tmp_path, [target], active_stages=["mapping"]
            )
            apply_post_create_run_setup(
                state, run_id, ctx.targets, ctx.cfg, ["mapping"]
            )
            self.assertFalse(ctx.cfg.skip_artifact_verify)
            label = target.label()

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
            ) as mock_launch, unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler._run_verify_pass",
                return_value=0,
            ) as mock_verify:
                _tick_run(state, run_id, ctx)
                mock_verify.assert_called()
                mock_launch.assert_not_called()

            tess = state.get_stage_run(run_id, label, "tess_ffi_download")
            self.assertEqual(tess.status, STATUS_EXTERNAL)
            mapping = state.get_stage_run(run_id, label, "mapping")
            self.assertEqual(mapping.status, STATUS_PENDING)


if __name__ == "__main__":
    unittest.main()
