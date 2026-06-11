"""End-to-end scheduler tick test across all seven pipeline stages."""

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

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.launcher import LaunchDescriptor
from syndiff_pipeline.common.orchestration.scheduler import (
    _apply_verify_outcome,
    _tick_run,
)
from syndiff_pipeline.common.orchestration.verify_worker import get_verify_worker
from syndiff_pipeline.common.orchestration.state import (
    STATUS_RUNNING,
    STATUS_SUCCESS,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.orchestration.verify_worker import reset_verify_worker_for_tests
from syndiff_pipeline.pipeline_spec import STAGE_DEPS
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


def _seven_stage_run_setup(tmp_path: Path, target: Target):
    """Run directory + state with all seven stages and local executors."""
    site = tmp_path / "site"
    site.mkdir()
    write_site_deployment(
        site,
        workspace_root=str(tmp_path),
        data_root=str(tmp_path / "data"),
    )
    _write_diff_policy(site / "diff_config.yaml")

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
                f"workspace_root: {tmp_path}",
                f"runs_root: {runs_root}",
                f"state_db_path: {tmp_path / 'state.sqlite'}",
                "skycell_wcs_csv: x.csv",
                f"diff_config_path: {site / 'diff_config.yaml'}",
                "stages:",
                "  mapping: {executor: local}",
                "  ps1_process: {executor: local}",
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

    all_stages = [
        "tess_ffi_download",
        "wcs_grouping",
        "mapping",
        "ps1_download",
        "ps1_process",
        "downsample",
        "diff",
    ]
    from syndiff_pipeline.common.orchestration.run_context import resolve_run_context
    from syndiff_pipeline.common.orchestration.state import PipelineState

    state = PipelineState(str(tmp_path / "state.sqlite"))
    state.create_run(
        run_id,
        str(cfg_path),
        str(run_dir / "targets.csv"),
        str(runs_root),
        [target],
        all_stages,
    )
    ctx = resolve_run_context(run_dir=run_dir)
    return state, ctx, run_id, runs_root


def _assert_topo_order(calls: list[str]) -> None:
    seen: set[str] = set()
    for stage in calls:
        for dep in STAGE_DEPS.get(stage, ()):
            if dep not in seen:
                raise AssertionError(f"{stage!r} launched before dependency {dep!r}")
        seen.add(stage)


class TestSevenStageTick(unittest.TestCase):
    def tearDown(self) -> None:
        reset_verify_worker_for_tests()

    def test_all_stages_progress_to_diff(self) -> None:
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        all_stages = [
            "tess_ffi_download",
            "wcs_grouping",
            "mapping",
            "ps1_download",
            "ps1_process",
            "downsample",
            "diff",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _seven_stage_run_setup(tmp_path, target)
            label = target.label()
            launch_calls: list[str] = []

            def fake_launch(
                _cmd,
                *,
                cfg,
                stage,
                runs_root,
                run_id,
                target_label,
                launch_token,
                **_kwargs,
            ):
                launch_calls.append(stage)
                logs.write_json_atomic(
                    logs.stage_status_path(runs_root, run_id, target_label, stage),
                    {
                        "launch_token": launch_token,
                        "pid": 424242,
                        "state": "exited",
                        "exit_code": 0,
                        "started_at": logs._utc_now_iso(),
                        "finished_at": logs._utc_now_iso(),
                        "updated_at": logs._utc_now_iso(),
                    },
                )
                return LaunchDescriptor(
                    executor="local",
                    native_id=424242,
                    launch_token=launch_token,
                    submit_epoch=0.0,
                )

            def drain_verify() -> None:
                get_verify_worker().drain(
                    lambda outcome: _apply_verify_outcome(state, outcome),
                    run_id=run_id,
                    block=True,
                    block_timeout_s=5.0,
                )

            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.verify_worker.stage_complete",
                return_value=False,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=fake_launch,
            ):
                for _ in range(20):
                    _tick_run(state, run_id, ctx)
                    drain_verify()
                    diff_row = state.get_stage_run(run_id, label, "diff")
                    if diff_row.status in (STATUS_RUNNING, STATUS_SUCCESS):
                        break

            diff_row = state.get_stage_run(run_id, label, "diff")
            self.assertIn(
                diff_row.status,
                (STATUS_RUNNING, STATUS_SUCCESS),
                f"diff stage status was {diff_row.status!r}",
            )
            self.assertGreater(len(launch_calls), 0)
            _assert_topo_order(launch_calls)
            template_calls = [s for s in launch_calls if s != "diff"]
            self.assertEqual(template_calls, [s for s in all_stages if s != "diff"][: len(template_calls)])


if __name__ == "__main__":
    unittest.main()
