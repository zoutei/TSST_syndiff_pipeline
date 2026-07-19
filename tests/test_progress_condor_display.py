"""Tests for Condor idle/running/held display in syndiff progress detail."""

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
from syndiff_pipeline.common.orchestration.state import PipelineState, STATUS_RUNNING
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.run_report import format_progress_lines
from tests.site_fixtures import write_site_config


def _condor_running_row(
    state: PipelineState,
    *,
    run_id: str,
    runs_root: str,
    stage: str = "diff",
    native_id: int | None = 42,
    executor: str = "condor",
) -> str:
    target = Target(20, 3, 3, 221.0, 38.0, "2020ut")
    state.create_run(run_id, "/cfg.yaml", "/targets.csv", runs_root, [target], [stage])
    label = target.label()
    state.update_stage_status(run_id, label, stage, STATUS_RUNNING, started_at="t")
    state.set_launch_descriptor(
        run_id,
        label,
        stage,
        executor=executor,
        native_id=native_id,
        submit_epoch=0.0,
        log_path=None,
    )
    return label


class TestCondorStatusLabels(unittest.TestCase):
    def test_condor_status_label_maps_idle_running_held(self):
        self.assertEqual(condor.condor_status_label(condor._JOB_IDLE), "idle")
        self.assertEqual(condor.condor_status_label(condor._JOB_RUNNING), "running")
        self.assertEqual(condor.condor_status_label(condor._JOB_HELD), "held")
        self.assertIsNone(condor.condor_status_label(None))
        self.assertIsNone(condor.condor_status_label(99))

    def test_format_condor_job_suffix(self):
        self.assertEqual(
            condor.format_condor_job_suffix(3, condor._JOB_IDLE),
            "condor idle c3.0",
        )
        self.assertEqual(condor.format_condor_job_suffix(3, None), "")


class TestProgressCondorDisplay(unittest.TestCase):
    def test_condor_idle_in_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            _condor_running_row(state, run_id="run_a", runs_root=runs_root)
            with mock.patch.object(
                condor, "query_clusters_display", return_value={42: (condor._JOB_IDLE, None)}
            ):
                lines = format_progress_lines(state, "run_a", runs_root)
            detail = [line for line in lines if line.startswith("  s")]
            self.assertEqual(len(detail), 1)
            self.assertIn("condor idle c42.0", detail[0])
            self.assertIn("(no log progress yet)", detail[0])

    def test_condor_running_appended_to_log_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            run_dir = runs_root / "run_a" / "per_target" / "s0020_c3_k3_2020ut"
            run_dir.mkdir(parents=True)
            log_path = run_dir / "diff.log"
            log_path.write_text("epsf_r1 12/48\n", encoding="utf-8")

            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            label = _condor_running_row(state, run_id="run_a", runs_root=str(runs_root))
            state.set_launch_descriptor(
                "run_a",
                label,
                "diff",
                executor="condor",
                native_id=2,
                submit_epoch=0.0,
                log_path=str(log_path),
            )
            with mock.patch.object(
                condor, "query_clusters_display", return_value={2: (condor._JOB_RUNNING, None)}
            ):
                lines = format_progress_lines(state, "run_a", str(runs_root))
            detail = [line for line in lines if "epsf" in line or "diff:" in line]
            self.assertTrue(any("condor running c2.0" in line for line in detail))

    def test_condor_held_in_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            _condor_running_row(state, run_id="run_a", runs_root=runs_root, native_id=7)
            with mock.patch.object(
                condor, "query_clusters_display", return_value={7: (condor._JOB_HELD, None)}
            ):
                lines = format_progress_lines(state, "run_a", runs_root)
            self.assertTrue(any("condor held c7.0" in line for line in lines))

    def test_local_running_has_no_condor_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            _condor_running_row(
                state,
                run_id="run_a",
                runs_root=runs_root,
                stage="mapping",
                native_id=99,
                executor="local",
            )
            with mock.patch.object(condor, "query_clusters_display") as query:
                lines = format_progress_lines(state, "run_a", runs_root)
            query.assert_not_called()
            detail = [line for line in lines if line.startswith("  s")]
            self.assertNotIn("condor", detail[0])

    def test_condor_unsubmitted_when_native_id_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            _condor_running_row(
                state,
                run_id="run_a",
                runs_root=runs_root,
                native_id=None,
            )
            with mock.patch.object(condor, "query_clusters_display") as query:
                lines = format_progress_lines(state, "run_a", runs_root)
            query.assert_not_called()
            self.assertTrue(any("condor unsubmitted" in line for line in lines))

    def test_condor_query_failure_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            _condor_running_row(state, run_id="run_a", runs_root=runs_root)
            with mock.patch.object(condor, "query_clusters_display", side_effect=OSError("no condor")):
                lines = format_progress_lines(state, "run_a", runs_root)
            detail = [line for line in lines if line.startswith("  s")]
            self.assertIn("(no log progress yet)", detail[0])
            self.assertNotIn("condor", detail[0])

    def test_no_detail_skips_condor_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = str(Path(tmp) / "runs")
            state = PipelineState(str(Path(tmp) / "state.sqlite"))
            _condor_running_row(state, run_id="run_a", runs_root=runs_root)
            with mock.patch.object(condor, "query_clusters_display") as query:
                lines = format_progress_lines(
                    state, "run_a", runs_root, include_running_detail=False
                )
            query.assert_not_called()
            self.assertTrue(all(not line.startswith("  s") for line in lines))

    def test_infers_condor_executor_from_frozen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handoff = tmp_path / "handoff"
            runs_root = handoff / "runs"
            run_dir = runs_root / "run_a"
            run_dir.mkdir(parents=True)
            write_site_config(
                run_dir / "config.yaml",
                workspace_root=str(handoff),
                data_root=str(tmp_path / "data"),
            )
            (run_dir / "config.yaml").write_text(
                (run_dir / "config.yaml").read_text(encoding="utf-8")
                + "\nstages:\n  star:\n    executor: condor\n",
                encoding="utf-8",
            )

            state = PipelineState(str(tmp_path / "state.sqlite"))
            target = Target(20, 3, 2, 0.0, 0.0, "s20_astrometry")
            state.create_run(
                "run_a",
                str(run_dir / "config.yaml"),
                "/targets.csv",
                str(runs_root),
                [target],
                ["star"],
            )
            label = target.label()
            state.update_stage_status("run_a", label, "star", STATUS_RUNNING, started_at="t")
            state.set_launch_descriptor(
                "run_a",
                label,
                "star",
                executor="condor",
                native_id=3,
                submit_epoch=0.0,
                log_path=None,
            )
            with mock.patch.object(
                condor, "query_clusters_display", return_value={3: (condor._JOB_IDLE, None)}
            ) as query:
                lines = format_progress_lines(state, "run_a", str(runs_root))
            query.assert_called_once()
            self.assertTrue(any("condor idle c3.0" in line for line in lines))


class TestStageShortNamesLightweight(unittest.TestCase):
    def test_stage_short_names_avoids_heavy_stage_imports(self):
        import sys

        heavy = (
            "syndiff_pipeline.difference_imaging.orchestration.stages",
            "syndiff_pipeline.star.orchestration.stages",
        )
        before = set(sys.modules)
        from syndiff_pipeline.pipeline_spec import stage_short_names

        names = stage_short_names()
        for mod in heavy:
            if mod not in before:
                self.assertNotIn(mod, sys.modules)
        self.assertEqual(names["downsample"], "down")
        self.assertEqual(names["diff"], "diff")
        self.assertEqual(names["star"], "star")


if __name__ == "__main__":
    unittest.main()
