"""Tests for Phase 0.6 small orchestrator fixes."""
from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.scheduler import reconcile_running_stages
from syndiff_pipeline.common.orchestration.state import (
    RUN_CANCELED,
    RUN_FAILED,
    STATUS_CANCELED,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_RUNNING,
    PipelineState,
    derive_run_final_status,
)
from syndiff_pipeline.common.orchestration.targets import Target, find_target
from tests.test_scheduler_recovery import _minimal_run


class TestDeriveRunFinalStatus(unittest.TestCase):
    def test_failed_outranks_canceled(self):
        counts = {STATUS_FAILED: 1, STATUS_CANCELED: 2}
        self.assertEqual(derive_run_final_status(counts), RUN_FAILED)

    def test_canceled_when_no_failed(self):
        counts = {STATUS_CANCELED: 1}
        self.assertEqual(derive_run_final_status(counts), RUN_CANCELED)


class TestFindTarget(unittest.TestCase):
    def test_accepts_full_label(self):
        targets = [
            Target(22, 3, 3, 228.0, 52.0, "2020dgc"),
            Target(22, 3, 3, 229.0, 53.0, "2020xyz"),
        ]
        t = find_target(targets, "s0022_c3_k3_2020dgc")
        self.assertEqual(t.target_name, "2020dgc")

    def test_scc_ambiguous_raises(self):
        targets = [
            Target(22, 3, 3, 228.0, 52.0, "2020dgc"),
            Target(22, 3, 3, 229.0, 53.0, "2020xyz"),
        ]
        with self.assertRaisesRegex(KeyError, "ambiguous"):
            find_target(targets, "22,3,3")


class TestNullExitCode(unittest.TestCase):
    def test_null_exit_code_dead_pid_requeues(self):
        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
            label = target.label()
            runs_root = str(tmp_path / "runs")
            token = state.new_launch_token()
            state.update_stage_status("run_a", label, "ps1_download", "ready")
            state.try_atomic_claim(
                "run_a",
                label,
                "ps1_download",
                launch_token=token,
                executor="local",
                native_id=12345,
                log_path=str(tmp_path / "x.log"),
            )
            from syndiff_pipeline.common.orchestration import logs
            from syndiff_pipeline.common.orchestration.run_context import resolve_run_context

            status_path = logs.stage_status_path(runs_root, "run_a", label, "ps1_download")
            logs.write_json_atomic(
                status_path,
                {
                    "launch_token": token,
                    "pid": 12345,
                    "state": "exited",
                    "exit_code": None,
                    "started_at": "t0",
                    "finished_at": "t1",
                },
            )
            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.daemon.is_process_alive",
                return_value=False,
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_download")
            self.assertEqual(counts["requeued"], 1)
            self.assertEqual(counts["still_running"], 0)
            self.assertEqual(row.status, STATUS_READY)

    def test_null_exit_code_live_pid_stays_running(self):
        target = Target(40, 1, 1, 292.6, 35.7, "2021udg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, run_dir = _minimal_run(tmp_path, target, ["ps1_download"])
            label = target.label()
            runs_root = str(tmp_path / "runs")
            token = state.new_launch_token()
            state.update_stage_status("run_a", label, "ps1_download", "ready")
            state.try_atomic_claim(
                "run_a",
                label,
                "ps1_download",
                launch_token=token,
                executor="local",
                native_id=12345,
                log_path=str(tmp_path / "x.log"),
            )
            from syndiff_pipeline.common.orchestration import logs
            from syndiff_pipeline.common.orchestration.run_context import resolve_run_context

            status_path = logs.stage_status_path(runs_root, "run_a", label, "ps1_download")
            logs.write_json_atomic(
                status_path,
                {
                    "launch_token": token,
                    "pid": 12345,
                    "state": "exited",
                    "exit_code": None,
                    "started_at": "t0",
                    "finished_at": "t1",
                },
            )
            ctx = resolve_run_context(run_dir=run_dir)
            with unittest.mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.daemon.is_process_alive",
                return_value=True,
            ):
                counts = reconcile_running_stages(state, "run_a", ctx)
            row = state.get_stage_run("run_a", label, "ps1_download")
            self.assertEqual(counts["still_running"], 1)
            self.assertEqual(row.status, STATUS_RUNNING)


if __name__ == "__main__":
    unittest.main()
