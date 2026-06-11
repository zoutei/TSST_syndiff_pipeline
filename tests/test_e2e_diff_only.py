"""End-to-end diff-only run: external upstream verify then diff launch."""

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

from syndiff_pipeline.common.orchestration.scheduler import _tick_run
from syndiff_pipeline.common.orchestration.run_setup import apply_post_create_run_setup
from syndiff_pipeline.common.orchestration.state import (
    SKIP_REASON_NOT_SELECTED,
    STATUS_EXTERNAL,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SKIPPED,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.orchestration.verify_worker import reset_verify_worker_for_tests
from syndiff_pipeline.common.orchestration.launcher import LaunchDescriptor
from tests.site_fixtures import write_site_deployment


def _write_diff_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "deployment_file: deployment.yaml",
                "defaults:",
                "  n_jobs: 2",
                "paths:",
                "  template_base: shifted_downsampled",
                "pipeline:",
                "  - kind: shared_mask",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _diff_only_run_setup(tmp_path: Path, target: Target, *, run_id: str = "run_a"):
    site = tmp_path / "site"
    site.mkdir()
    write_site_deployment(
        site,
        workspace_root=str(tmp_path),
        data_root=str(tmp_path / "data"),
    )
    _write_diff_policy(site / "diff_config.yaml")

    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "per_target").mkdir()
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"data_root: {tmp_path / 'data'}",
                f"workspace_root: {tmp_path}",
                f"runs_root: {runs_root}",
                f"state_db_path: {tmp_path / 'state.sqlite'}",
                "skycell_wcs_csv: x.csv",
                f"diff_config_path: {site / 'diff_config.yaml'}",
                "stages:",
                "  diff: {executor: local}",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "targets.csv").write_text(
        "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
        f"{target.sector},{target.camera},{target.ccd},"
        f"{target.target_ra},{target.target_dec},{target.target_name},true\n",
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_diff_config_path": str(site / "diff_config.yaml"),
            }
        ),
        encoding="utf-8",
    )

    from syndiff_pipeline.common.orchestration.run_context import resolve_run_context
    from syndiff_pipeline.common.orchestration.state import PipelineState

    state = PipelineState(str(tmp_path / "state.sqlite"))
    state.create_run(
        run_id,
        str(cfg_path),
        str(run_dir / "targets.csv"),
        str(runs_root),
        [target],
        ["diff"],
    )
    ctx = resolve_run_context(run_dir=run_dir)
    apply_post_create_run_setup(state, run_id, ctx.targets, ctx.cfg, ["diff"])
    return state, ctx, run_id


DIFF_VERIFY_STAGES = (
    "tess_ffi_download",
    "wcs_grouping",
    "downsample",
)
DIFF_NOT_SELECTED_STAGES = (
    "mapping",
    "ps1_download",
    "ps1_process",
)


class TestDiffOnlyE2E(unittest.TestCase):
    def tearDown(self) -> None:
        reset_verify_worker_for_tests()

    def test_diff_only_marks_mapping_and_ps1_not_selected(self) -> None:
        target = Target(20, 3, 3, 210.219333, 81.846589, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, _ctx, run_id = _diff_only_run_setup(tmp_path, target)
            label = target.label()

            for stage in DIFF_VERIFY_STAGES:
                self.assertEqual(
                    state.get_stage_run(run_id, label, stage).status,
                    STATUS_EXTERNAL,
                )
            for stage in DIFF_NOT_SELECTED_STAGES:
                row = state.get_stage_run(run_id, label, stage)
                self.assertEqual(row.status, STATUS_SKIPPED)
                self.assertEqual(
                    state.get_skip_reason(run_id, label, stage),
                    SKIP_REASON_NOT_SELECTED,
                )

    def test_external_upstream_skipped_then_diff_launches(self) -> None:
        target = Target(20, 3, 3, 210.219333, 81.846589, "2020ut")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id = _diff_only_run_setup(tmp_path, target)
            label = target.label()

            for stage in DIFF_VERIFY_STAGES:
                row = state.get_stage_run(run_id, label, stage)
                self.assertEqual(row.status, STATUS_EXTERNAL)

            launch_calls: list[str] = []

            def fake_launch(*_args, **kwargs):
                launch_calls.append(kwargs["stage"])
                return LaunchDescriptor(
                    executor="local",
                    native_id=12345,
                    launch_token=kwargs["launch_token"],
                    submit_epoch=0.0,
                )

            def complete(_resolved, stage, **_kwargs):
                return stage in DIFF_VERIFY_STAGES

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                side_effect=complete,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.reconcile_running_stages",
                return_value={},
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=fake_launch,
            ):
                for _ in range(12):
                    _tick_run(state, run_id, ctx)
                    if launch_calls:
                        break
                for _ in range(4):
                    _tick_run(state, run_id, ctx)

            for stage in DIFF_VERIFY_STAGES:
                self.assertEqual(
                    state.get_stage_run(run_id, label, stage).status,
                    STATUS_SKIPPED,
                )
            for stage in DIFF_NOT_SELECTED_STAGES:
                self.assertEqual(
                    state.get_skip_reason(run_id, label, stage),
                    SKIP_REASON_NOT_SELECTED,
                )
            diff_row = state.get_stage_run(run_id, label, "diff")
            self.assertIn(diff_row.status, (STATUS_READY, STATUS_RUNNING))
            self.assertEqual(launch_calls, ["diff"])


if __name__ == "__main__":
    unittest.main()
