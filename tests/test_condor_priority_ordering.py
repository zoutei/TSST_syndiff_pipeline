"""Tests for CSV-row-order-derived Condor job priority.

First row in a submitted --scc/--targets CSV is the most important target;
that ordering is expressed as each target's Condor `priority` (-> `JobPrio`
in the live ClassAd) submit-file attribute, applied uniformly to every stage
of that target. See PipelineState.create_run, launcher.launch_stage, and
condor.write_submit_file.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import condor
from syndiff_pipeline.common.orchestration.scheduler import _try_launch_ready_row
from syndiff_pipeline.common.orchestration.state import STATUS_READY, PipelineState
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.runner_config import RunnerConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import parse_stage_params
from tests.test_daemon_behavior import _minimal_run_setup


def _target(sector: int, name: str) -> Target:
    return Target(
        sector=sector,
        camera=3,
        ccd=3,
        target_ra=228.0,
        target_dec=52.0,
        target_name=name,
    )


class TestCreateRunPriority(unittest.TestCase):
    def test_priority_descends_with_csv_row_order(self):
        targets = [_target(20, "first"), _target(21, "second"), _target(22, "third")]
        stages = ["tess_ffi_download", "mapping"]
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, targets, stages)

            expected = {
                targets[0].label(): 0,
                targets[1].label(): -1,
                targets[2].label(): -2,
            }
            for target in targets:
                for stage in stages:
                    row = state.get_stage_run("run_a", target.label(), stage)
                    self.assertEqual(
                        row.priority,
                        expected[target.label()],
                        f"{target.label()}/{stage} priority mismatch",
                    )

    def test_every_stage_of_one_target_shares_its_priority(self):
        targets = [_target(20, "first"), _target(21, "second")]
        stages = ["tess_ffi_download", "mapping", "downsample"]
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, targets, stages)

            # Every canonical stage in the full DAG (not just the ones
            # selected to run), since create_run materializes a row per
            # pipeline_spec.stage_names regardless of `stages`.
            all_stage_names = state.pipeline_spec.stage_names
            second_priorities = {
                state.get_stage_run("run_a", targets[1].label(), s).priority
                for s in all_stage_names
            }
            self.assertEqual(second_priorities, {-1})

    def test_backfill_missing_stage_rows_inherits_target_priority(self):
        # Mirrors tests/test_backfill_stage_rows.py's delete-then-backfill
        # pattern: simulate a stage_runs row missing for one canonical stage
        # (e.g. a stage added to the pipeline after this run was created)
        # and confirm backfill_missing_stage_rows re-inserts it carrying the
        # target's existing priority rather than silently defaulting to 0.
        targets = [_target(20, "first"), _target(21, "second")]
        stages = ["tess_ffi_download", "mapping", "diff"]
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            state.create_run("run_a", "/cfg.yaml", "/targets.csv", tmp, targets, stages)
            with state._conn() as conn:
                conn.execute(
                    "DELETE FROM stage_runs WHERE run_id = ? AND stage = ?",
                    ("run_a", "diff"),
                )

            self.assertIsNone(state.get_stage_run("run_a", targets[0].label(), "diff"))
            inserted = state.backfill_missing_stage_rows("run_a")
            self.assertEqual(inserted, 2)

            first_diff = state.get_stage_run("run_a", targets[0].label(), "diff")
            second_diff = state.get_stage_run("run_a", targets[1].label(), "diff")
            self.assertEqual(first_diff.priority, 0)
            self.assertEqual(second_diff.priority, -1)


class TestWriteSubmitFilePriority(unittest.TestCase):
    def _artifacts(self, tmp: str) -> dict:
        return {
            "stdout": Path(tmp) / "out",
            "stderr": Path(tmp) / "err",
            "log": Path(tmp) / "log",
        }

    def test_default_priority_emitted_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            submit_path = Path(tmp) / "job.submit"
            condor.write_submit_file(
                submit_path, ["echo", "hi"], self._artifacts(tmp), condor.CondorResourceRequest()
            )
            self.assertIn("priority = 0", submit_path.read_text(encoding="utf-8"))

    def test_explicit_priority_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            submit_path = Path(tmp) / "job.submit"
            condor.write_submit_file(
                submit_path,
                ["echo", "hi"],
                self._artifacts(tmp),
                condor.CondorResourceRequest(priority=7),
            )
            self.assertIn("priority = 7", submit_path.read_text(encoding="utf-8"))


class TestLaunchStagePriority(unittest.TestCase):
    def test_launch_stage_overrides_static_profile_priority(self):
        from syndiff_pipeline.common.orchestration.launcher import launch_stage

        cfg = RunnerConfig(
            stages=parse_stage_params({"downsample": {"executor": "condor"}}),
            workspace_root="/tmp/ws",
            data_root="/tmp/data",
        )
        with mock.patch(
            "syndiff_pipeline.common.orchestration.launcher.condor.submit_job",
            return_value=(99, 1.0),
        ) as submit_job:
            launch_stage(
                ["python", "-m", "syndiff_pipeline.common.orchestration.run_stage"],
                cfg=cfg,
                stage="downsample",
                runs_root="/runs",
                run_id="run_a",
                target_label="s0020_c3_k3_2020ut",
                launch_token="tok",
                priority=5,
            )
        resources = submit_job.call_args.kwargs["resources"]
        self.assertEqual(resources.priority, 5)

    def test_launch_stage_overrides_resources_override_priority(self):
        from syndiff_pipeline.common.orchestration.launcher import launch_stage

        cfg = RunnerConfig(
            stages=parse_stage_params({"ps1_process": {"executor": "condor"}}),
            workspace_root="/tmp/ws",
            data_root="/tmp/data",
        )
        override = condor.CondorResourceRequest(request_cpus=16, request_memory_mb=25000)
        with mock.patch(
            "syndiff_pipeline.common.orchestration.launcher.condor.submit_job",
            return_value=(100, 1.0),
        ) as submit_job:
            launch_stage(
                ["python", "-m", "syndiff_pipeline.common.orchestration.run_stage"],
                cfg=cfg,
                stage="ps1_process",
                runs_root="/runs",
                run_id="run_a",
                target_label="s0020_c3_k3_2020ut",
                launch_token="tok",
                resources_override=override,
                priority=9,
            )
        resources = submit_job.call_args.kwargs["resources"]
        self.assertEqual(resources.priority, 9)
        # The rest of the override profile must survive the priority overlay.
        self.assertEqual(resources.request_cpus, 16)
        self.assertEqual(resources.request_memory_mb, 25000)

    def test_launch_stage_default_priority_is_zero(self):
        from syndiff_pipeline.common.orchestration.launcher import launch_stage

        cfg = RunnerConfig(
            stages=parse_stage_params({"downsample": {"executor": "condor"}}),
            workspace_root="/tmp/ws",
            data_root="/tmp/data",
        )
        with mock.patch(
            "syndiff_pipeline.common.orchestration.launcher.condor.submit_job",
            return_value=(99, 1.0),
        ) as submit_job:
            launch_stage(
                ["python", "-m", "syndiff_pipeline.common.orchestration.run_stage"],
                cfg=cfg,
                stage="downsample",
                runs_root="/runs",
                run_id="run_a",
                target_label="s0020_c3_k3_2020ut",
                launch_token="tok",
            )
        resources = submit_job.call_args.kwargs["resources"]
        self.assertEqual(resources.priority, 0)


class TestSchedulerThreadsRowPriorityToLaunch(unittest.TestCase):
    """Closes the gap between create_run's DB priority and launcher.launch_stage:
    deleting `priority=row.priority or 0` from scheduler.py's
    `_try_launch_ready_row` would leave every other test in this module green
    (they all call launch_stage directly with a literal priority), so this
    test exercises the actual scheduler->launcher call the daemon makes.
    """

    def test_try_launch_ready_row_passes_target_priority_to_launch_stage(self):
        targets = [_target(20, "first"), _target(21, "second")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state, ctx, run_id, runs_root = _minimal_run_setup(
                tmp_path, targets, active_stages=["tess_ffi_download"]
            )
            for t in targets:
                state.update_stage_status(run_id, t.label(), "tess_ffi_download", STATUS_READY)

            from syndiff_pipeline.common.orchestration.launcher import LaunchDescriptor

            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.launcher.launch_stage",
                side_effect=lambda *a, **kw: LaunchDescriptor(
                    executor="local", native_id=1, launch_token=kw.get("launch_token", "tok")
                ),
            ) as launch_stage:
                for t in targets:
                    row = state.get_stage_run(run_id, t.label(), "tess_ffi_download")
                    _try_launch_ready_row(
                        state,
                        run_id,
                        ctx,
                        row,
                        pool_label="network",
                        force_rerun=False,
                        active_stages=["tess_ffi_download"],
                        targets_by_label={t.label(): t for t in targets},
                        runs_root=str(runs_root),
                    )

            self.assertEqual(launch_stage.call_count, 2)
            priorities_by_label = {
                call.kwargs["target_label"]: call.kwargs["priority"]
                for call in launch_stage.call_args_list
            }
            self.assertEqual(
                priorities_by_label,
                {targets[0].label(): 0, targets[1].label(): -1},
            )


if __name__ == "__main__":
    unittest.main()
