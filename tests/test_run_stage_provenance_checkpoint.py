"""Verify run_stage's provenance checkpoint dual-write (BK-3).

Mirrors the mocking pattern in ``test_run_stage_downsample_progress.py``:
``resolve_run_context``/``resolve_config``/``dispatch.execute_stage``/
``collect_stage_artifacts``/``write_manifest`` are all mocked so this stays a
unit test of ``run_stage.main``'s control flow, not an integration test of
the real stage machinery.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CHECKPOINT_STAGES = {
    "tess_ffi_download": "emit_ffi_set_checkpoint",
    "mapping": "emit_mapping_checkpoint",
    "remap": "emit_remap_store_checkpoint",
    "downsample": "emit_downsample_checkpoint",
    "ps1_process": "emit_scc_assembly_checkpoint",
}


def _run_main(stage: str, *, emit_name: str, emit_side_effect=None):
    """Run ``run_stage.main`` for *stage* with the checkpoint emitter mocked.

    Returns ``(rc, emit_mock, fake_resolved)``.
    """
    fake_ctx = mock.Mock()
    fake_ctx.cfg.runs_dir.return_value = "/runs"
    fake_ctx.targets = [mock.Mock(label=mock.Mock(return_value="t1"))]
    fake_resolved = mock.Mock()

    def fake_execute_stage(resolved, s, force_rerun=False, *, progress_path=None):
        return None

    emit_patch = (
        "syndiff_pipeline.template_creation.orchestration."
        f"provenance_checkpoint.{emit_name}"
    )
    with mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.resolve_run_context",
        return_value=fake_ctx,
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.resolve_config",
        return_value=fake_resolved,
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.dispatch.execute_stage",
        side_effect=fake_execute_stage,
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
        return_value=(0, 0, []),
    ), mock.patch(
        "syndiff_pipeline.common.orchestration.run_stage.write_manifest",
    ), mock.patch(
        emit_patch,
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
    return rc, emit_mock, fake_resolved


class TestRunStageProvenanceCheckpoint(unittest.TestCase):
    def test_success_emits_checkpoint_after_manifest(self):
        for stage, emit_name in _CHECKPOINT_STAGES.items():
            with self.subTest(stage=stage):
                rc, emit_mock, fake_resolved = _run_main(stage, emit_name=emit_name)
                self.assertEqual(rc, 0)
                emit_mock.assert_called_once_with(fake_resolved)

    def test_stage_still_succeeds_when_checkpoint_emit_raises(self):
        stage = "downsample"
        emit_name = _CHECKPOINT_STAGES[stage]
        rc, emit_mock, _ = _run_main(
            stage,
            emit_name=emit_name,
            emit_side_effect=RuntimeError("provenance spool unwritable"),
        )
        self.assertEqual(rc, 0)
        emit_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
