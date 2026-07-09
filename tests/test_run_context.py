"""Tests for frozen run-local config and targets."""
from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import cli as orch_cli
from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.state import PipelineState, RUN_CANCELED, STATUS_PENDING
from syndiff_pipeline.common.orchestration.targets import Target, load_targets, write_normalized_targets
from syndiff_pipeline.template_creation.orchestration import dispatch
from syndiff_pipeline.common.orchestration.run_context import resolve_run_context, resolve_run_control_context
from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config
from tests.site_fixtures import write_site_config


def _write_targets(path: Path) -> None:
    path.write_text(
        "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
        "23,1,3,185.0,5.3,2020ftl,true\n",
        encoding="utf-8",
    )


def _write_star_format_targets(path: Path) -> None:
    path.write_text(
        "sector,camera,ccd,target_name,stars_file,baseline_workspace_run_id,"
        "baseline_diffs,baseline_convolved,phot_bkg,enabled\n"
        "20,3,2,s20_astrometry,star_hosts/example.csv,star_full_lc,hp_d,hp_c,ks_b_s,true\n",
        encoding="utf-8",
    )


def _prepare_run_dir(tmp_path: Path, *, run_name: str = "star_run") -> tuple[Path, Path]:
    handoff = tmp_path / "handoff"
    data = tmp_path / "data"
    source_cfg = tmp_path / "config.yaml"
    write_site_config(
        source_cfg,
        workspace_root=str(handoff),
        data_root=str(data),
    )
    run_dir = handoff / "runs" / run_name
    targets = tmp_path / "targets.csv"
    _write_targets(targets)
    logs.materialize_run_inputs(source_cfg, targets, run_dir)
    (run_dir / "run_meta.json").write_text(
        f'{{"run_id": "{run_name}"}}', encoding="utf-8"
    )
    return handoff, run_dir


class TestMaterializeRunInputs(unittest.TestCase):
    def test_relative_paths_normalized_to_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff = tmp_path / "handoff"
            data = tmp_path / "data"
            source_cfg = tmp_path / "site" / "config.yaml"
            write_site_config(
                source_cfg,
                workspace_root=str(handoff),
                data_root=str(data),
            )
            targets = tmp_path / "targets.csv"
            _write_targets(targets)

            run_dir = tmp_path / "runs" / "run_a"
            cfg_path, targets_path = logs.materialize_run_inputs(source_cfg, targets, run_dir)

            self.assertEqual(cfg_path, str(run_dir / "config.yaml"))
            self.assertEqual(targets_path, str(run_dir / "targets.csv"))
            frozen = load_runner_config(cfg_path)
            self.assertTrue(Path(frozen.data_root).is_absolute())
            self.assertTrue(Path(frozen.skycell_wcs_csv).is_absolute())

    def test_existing_frozen_copy_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff = tmp_path / "handoff"
            data = tmp_path / "data"
            source_cfg = tmp_path / "config.yaml"
            write_site_config(
                source_cfg,
                workspace_root=str(handoff),
                data_root=str(data),
            )
            targets = tmp_path / "targets.csv"
            _write_targets(targets)
            run_dir = tmp_path / "run_a"
            logs.materialize_run_inputs(source_cfg, targets, run_dir)

            frozen_cfg = run_dir / "config.yaml"
            frozen_cfg.write_text("data_root: /frozen\nworkspace_root: /frozen\n", encoding="utf-8")

            cfg_path, _ = logs.materialize_run_inputs(source_cfg, targets, run_dir)
            self.assertIn("/frozen", Path(cfg_path).read_text(encoding="utf-8"))


class TestResolveRunContext(unittest.TestCase):
    def test_resolve_from_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff = tmp_path / "handoff"
            data = tmp_path / "data"
            source_cfg = tmp_path / "config.yaml"
            write_site_config(
                source_cfg,
                workspace_root=str(handoff),
                data_root=str(data),
            )
            targets = tmp_path / "targets.csv"
            _write_targets(targets)
            run_dir = tmp_path / "run_a"
            logs.materialize_run_inputs(source_cfg, targets, run_dir)
            (run_dir / "run_meta.json").write_text(
                '{"run_id": "run_a"}', encoding="utf-8"
            )

            ctx = resolve_run_context(run_dir=run_dir)
            self.assertEqual(ctx.run_id, "run_a")
            self.assertEqual(len(ctx.targets), 1)
            self.assertEqual(ctx.targets[0].target_name, "2020ftl")

    def test_resolve_with_config_and_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff = tmp_path / "handoff"
            data = tmp_path / "data"
            runs_root = handoff / "runs"
            source_cfg = tmp_path / "config.yaml"
            write_site_config(
                source_cfg,
                workspace_root=str(handoff),
                data_root=str(data),
            )
            targets = tmp_path / "targets.csv"
            _write_targets(targets)
            run_dir = runs_root / "run_a"
            logs.materialize_run_inputs(source_cfg, targets, run_dir)

            ctx = resolve_run_context(run_id="run_a", runs_root=str(runs_root))
            self.assertEqual(ctx.run_id, "run_a")
            self.assertEqual(ctx.run_dir, run_dir.resolve())

    def test_resolve_control_context_skips_invalid_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, run_dir = _prepare_run_dir(tmp_path)
            (run_dir / "targets.csv").write_text(
                "sector,camera,ccd,target_name,enabled\n20,3,2,s20_astrometry,true\n",
                encoding="utf-8",
            )

            ctx = resolve_run_control_context(run_dir=run_dir)
            self.assertEqual(ctx.run_id, "star_run")
            self.assertEqual(ctx.cfg.workspace_root, str(tmp_path / "handoff"))

    def test_resolve_context_rejects_invalid_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, run_dir = _prepare_run_dir(tmp_path)
            (run_dir / "targets.csv").write_text(
                "sector,camera,ccd,target_name,enabled\n20,3,2,s20_astrometry,true\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                resolve_run_context(run_dir=run_dir)

    def test_resolve_control_context_by_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff, run_dir = _prepare_run_dir(tmp_path)
            _write_star_format_targets(run_dir / "targets.csv")

            ctx = resolve_run_control_context(
                run_id="star_run",
                runs_root=str(handoff / "runs"),
            )
            self.assertEqual(ctx.run_dir, run_dir.resolve())

    def test_resolve_control_context_missing_config_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "broken_run"
            run_dir.mkdir()
            (run_dir / "targets.csv").write_text("x\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                resolve_run_control_context(run_dir=run_dir)


class TestWriteNormalizedTargets(unittest.TestCase):
    def test_roundtrip_load_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.csv"
            rows = [
                Target(
                    sector=20,
                    camera=3,
                    ccd=2,
                    target_ra=12.5,
                    target_dec=-3.1,
                    target_name="s20_astrometry",
                    enabled=True,
                )
            ]
            write_normalized_targets(path, rows)
            loaded = load_targets(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].target_name, "s20_astrometry")
            self.assertAlmostEqual(loaded[0].target_ra, 12.5)
            self.assertAlmostEqual(loaded[0].target_dec, -3.1)


class TestRunStarTargetsPath(unittest.TestCase):
    def test_run_star_targets_path_under_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_x"
            run_dir.mkdir()
            expected = run_dir / "star_targets.csv"
            self.assertEqual(logs.run_star_targets_path(run_dir), expected)


class TestCmdKillControl(unittest.TestCase):
    def test_cmd_kill_works_with_invalid_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, run_dir = _prepare_run_dir(tmp_path)
            _write_star_format_targets(run_dir / "targets.csv")

            args = Namespace(run_dir=str(run_dir), run_id=None, deployment=None)
            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.warn_if_daemon_host_mismatch"
            ):
                with mock.patch(
                    "syndiff_pipeline.common.orchestration.cli.PipelineState"
                ) as mock_state_cls:
                    with mock.patch(
                        "syndiff_pipeline.common.orchestration.condor.sweep_run_condor_audit_clusters"
                    ) as mock_sweep:
                        mock_state = mock_state_cls.return_value
                        rc = orch_cli.cmd_kill(args)

            self.assertEqual(rc, 0)
            mock_state.insert_command.assert_called_once_with("cancel", run_id="star_run")
            mock_sweep.assert_called_once()

    def test_cmd_kill_by_run_id_with_invalid_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff, run_dir = _prepare_run_dir(tmp_path)
            _write_star_format_targets(run_dir / "targets.csv")

            deploy = tmp_path / "deployment.yaml"
            deploy.write_text(f"workspace_root: {handoff}\n", encoding="utf-8")
            args = Namespace(
                run_dir=None,
                run_id="star_run",
                deployment=str(deploy),
            )
            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.warn_if_daemon_host_mismatch"
            ):
                with mock.patch(
                    "syndiff_pipeline.common.orchestration.cli.PipelineState"
                ) as mock_state_cls:
                    with mock.patch(
                        "syndiff_pipeline.common.orchestration.condor.sweep_run_condor_audit_clusters"
                    ):
                        rc = orch_cli.cmd_kill(args)

            self.assertEqual(rc, 0)
            mock_state_cls.return_value.insert_command.assert_called_once_with(
                "cancel", run_id="star_run"
            )


class TestApplyCancelRun(unittest.TestCase):
    def test_cancel_pending_run_leaves_terminal_status(self):
        target = Target(20, 3, 2, 0.0, 0.0, "s20_astrometry")
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run("star_run", "/c", "/t", tmp, [target], ["star"])
            label = target.label()
            row = state.get_stage_run("star_run", label, "star")
            self.assertEqual(row.status, STATUS_PENDING)
            self.assertIn("star_run", [r["run_id"] for r in state.active_runs()])

            counts = state.apply_cancel_run("star_run")
            self.assertEqual(counts["canceled"], 0)
            self.assertEqual((state.get_run("star_run") or {}).get("status"), RUN_CANCELED)
            self.assertNotIn("star_run", [r["run_id"] for r in state.active_runs()])
            self.assertEqual(
                state.get_stage_run("star_run", label, "star").status,
                RUN_CANCELED,
            )


class TestBuildStageCommand(unittest.TestCase):
    def test_uses_run_dir_not_config_paths(self):
        cmd = dispatch.build_stage_command(
            "run_a",
            "mapping",
            "/handoff/runs/run_a",
            "s0023_c1_k3_2020ftl",
            launch_token="test-token",
        )
        self.assertIn("--run-dir", cmd)
        self.assertIn("/handoff/runs/run_a", cmd)
        self.assertNotIn("--config", cmd)
        self.assertNotIn("--targets", cmd)


if __name__ == "__main__":
    unittest.main()
