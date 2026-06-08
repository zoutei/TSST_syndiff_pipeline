"""Tests for frozen run-local config and targets."""
from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner import logs, stages
from syndiff_pipeline.template_runner.run_context import (
    RUNS_ROOT_ENV_VAR,
    resolve_run_context,
    runs_root_from_env,
)
from syndiff_pipeline.template_runner.runner_config import load_runner_config


def _write_minimal_config(path: Path, *, data_root: str = "/data") -> None:
    path.write_text(
        "\n".join(
            [
                f"data_root: {data_root}",
                "handoff_root: /handoff",
                "runs_root: /handoff/runs",
                "skycell_wcs_csv: skycells.csv",
            ]
        ),
        encoding="utf-8",
    )
    (path.parent / "skycells.csv").write_text("x", encoding="utf-8")


def _write_targets(path: Path) -> None:
    path.write_text(
        "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
        "23,1,3,185.0,5.3,2020ftl,true\n",
        encoding="utf-8",
    )


class TestMaterializeRunInputs(unittest.TestCase):
    def test_relative_paths_normalized_to_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_cfg = tmp_path / "site" / "config.yaml"
            source_cfg.parent.mkdir(parents=True)
            _write_minimal_config(source_cfg, data_root="relative_data")
            (source_cfg.parent / "relative_data").mkdir()
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
            source_cfg = tmp_path / "config.yaml"
            _write_minimal_config(source_cfg)
            targets = tmp_path / "targets.csv"
            _write_targets(targets)
            run_dir = tmp_path / "run_a"
            logs.materialize_run_inputs(source_cfg, targets, run_dir)

            frozen_cfg = run_dir / "config.yaml"
            frozen_cfg.write_text("data_root: /frozen\n", encoding="utf-8")

            cfg_path, _ = logs.materialize_run_inputs(source_cfg, targets, run_dir)
            self.assertIn("/frozen", Path(cfg_path).read_text(encoding="utf-8"))


class TestRunsRootFromEnv(unittest.TestCase):
    def test_returns_none_when_unset(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(runs_root_from_env())

    def test_reads_syndiff_runs_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.dict("os.environ", {RUNS_ROOT_ENV_VAR: tmp}):
                self.assertEqual(runs_root_from_env(), Path(tmp).resolve())

    def test_resolve_with_env_and_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_cfg = tmp_path / "config.yaml"
            _write_minimal_config(source_cfg)
            targets = tmp_path / "targets.csv"
            _write_targets(targets)
            runs_root = tmp_path / "runs"
            run_dir = runs_root / "run_a"
            logs.materialize_run_inputs(source_cfg, targets, run_dir)

            with unittest.mock.patch.dict("os.environ", {RUNS_ROOT_ENV_VAR: str(runs_root)}):
                ctx = resolve_run_context(run_id="run_a", runs_root=str(runs_root))
            self.assertEqual(ctx.run_id, "run_a")
            self.assertEqual(ctx.run_dir, run_dir.resolve())


class TestResolveRunContext(unittest.TestCase):
    def test_resolve_from_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_cfg = tmp_path / "config.yaml"
            _write_minimal_config(source_cfg)
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


class TestBuildStageCommand(unittest.TestCase):
    def test_uses_run_dir_not_config_paths(self):
        cmd = stages.build_stage_command(
            "run_a",
            "mapping",
            "/handoff/runs/run_a",
            "s0023_c1_k3_2020ftl",
        )
        self.assertIn("--run-dir", cmd)
        self.assertIn("/handoff/runs/run_a", cmd)
        self.assertNotIn("--config", cmd)
        self.assertNotIn("--targets", cmd)


if __name__ == "__main__":
    unittest.main()
