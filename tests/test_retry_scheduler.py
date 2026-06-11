"""Tests for retry command intents and daemon ensure."""
from __future__ import annotations

import argparse
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
from syndiff_pipeline.template_creation.orchestration.cli import cmd_retry
from syndiff_pipeline.common.orchestration.run_context import RunContext, resolve_run_context
from syndiff_pipeline.common.orchestration.scheduler_control import ensure_daemon_running
from syndiff_pipeline.common.orchestration.workspace import record_deployment_path, state_db_path
from syndiff_pipeline.common.orchestration.state import (
    PipelineState,
    STATUS_FAILED,
)
from syndiff_pipeline.common.orchestration.targets import Target, find_target_for_run
from tests.site_fixtures import write_site_config


def _write_targets(path: Path, *, enabled: str = "true") -> None:
    path.write_text(
        "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
        f"23,1,3,185.0,5.3,2020ftl,{enabled}\n",
        encoding="utf-8",
    )


def _make_run_context(tmp: Path, *, enabled: str = "true") -> tuple[RunContext, PipelineState]:
    handoff = tmp / "handoff"
    data = tmp / "data"
    runs_root = handoff / "runs"
    state_db = state_db_path(handoff)
    source_cfg = tmp / "site" / "config.yaml"
    write_site_config(
        source_cfg,
        workspace_root=str(handoff),
        data_root=str(data),
    )
    targets_csv = tmp / "targets.csv"
    _write_targets(targets_csv, enabled=enabled)

    run_id = "run_a"
    run_dir = runs_root / run_id
    logs.materialize_run_inputs(source_cfg, targets_csv, run_dir)
    meta = {
        "run_id": run_id,
        "stages": ["mapping", "downsample"],
        "force_rerun": False,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    target = Target(
        sector=23,
        camera=1,
        ccd=3,
        target_ra=185.0,
        target_dec=5.3,
        target_name="2020ftl",
    )
    state = PipelineState(str(state_db))
    state.create_run(
        run_id,
        str(logs.run_config_path(run_dir)),
        str(logs.run_targets_path(run_dir)),
        str(runs_root),
        [target],
        ["mapping", "downsample"],
    )
    ctx = resolve_run_context(run_dir=run_dir)
    deploy_path = tmp / "site" / "deployment.yaml"
    if deploy_path.is_file():
        record_deployment_path(handoff, deploy_path)
    return ctx, state


class TestFindTargetForRun(unittest.TestCase):
    def test_falls_back_to_db_when_csv_row_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx, state = _make_run_context(Path(tmp), enabled="false")
            t = find_target_for_run(ctx, state, "23,1,3")
            self.assertEqual(t.label(), "s0023_c1_k3_2020ftl")


class TestEnsureDaemonRunning(unittest.TestCase):
    def test_spawns_when_no_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx, _state = _make_run_context(Path(tmp))
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.spawn_detached_daemon",
                return_value=4242,
            ) as spawn, unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.wait_for_daemon",
                return_value=True,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.read_process_identity",
                return_value=("localhost", 4242),
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.local_hostname",
                return_value="localhost",
            ):
                result = ensure_daemon_running(ctx.cfg.workspace_root)
            self.assertTrue(result.spawned)
            self.assertEqual(result.pid, 4242)
            spawn.assert_called_once()

    def test_does_not_spawn_when_daemon_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx, _state = _make_run_context(Path(tmp))
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon_is_alive",
                return_value=True,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control._supervisor_pid_identity",
                return_value=(None, 99999),
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.get_supervisor_host",
                return_value=None,
            ), unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon.spawn_detached_daemon",
            ) as spawn:
                result = ensure_daemon_running(ctx.cfg.workspace_root)
            self.assertFalse(result.spawned)
            self.assertEqual(result.pid, 99999)
            spawn.assert_not_called()


class TestCmdRetry(unittest.TestCase):
    def test_bulk_retry_inserts_command_and_ensures_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx, state = _make_run_context(Path(tmp))
            label = "s0023_c1_k3_2020ftl"
            state.update_stage_status(ctx.run_id, label, "mapping", STATUS_FAILED)
            args = argparse.Namespace(
                run_dir=str(ctx.run_dir),
                run_id=ctx.run_id,
                config=None,
                scc=None,
                stage=None,
                no_start_daemon=False,
            )
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.cli.ensure_daemon_running",
            ) as ensure:
                rc = cmd_retry(args)
            self.assertEqual(rc, 0)
            cmds = state.fetch_pending_commands()
            self.assertEqual(len(cmds), 1)
            self.assertEqual(cmds[0].kind, "retry")
            ensure.assert_called_once()
            self.assertEqual(ensure.call_args.args[0], ctx.cfg.workspace_root)
            self.assertIsNotNone(ensure.call_args.kwargs.get("deployment_path"))


if __name__ == "__main__":
    unittest.main()
