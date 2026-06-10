"""Tests for frozen run-local config and targets."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration import dispatch, logs
from syndiff_pipeline.template_creation.orchestration.run_context import resolve_run_context
from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config
from tests.site_config import write_site_config


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
            handoff = tmp_path / "handoff"
            data = tmp_path / "data"
            source_cfg = tmp_path / "site" / "config.yaml"
            write_site_config(
                source_cfg,
                handoff_root=str(handoff),
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
                handoff_root=str(handoff),
                data_root=str(data),
            )
            targets = tmp_path / "targets.csv"
            _write_targets(targets)
            run_dir = tmp_path / "run_a"
            logs.materialize_run_inputs(source_cfg, targets, run_dir)

            frozen_cfg = run_dir / "config.yaml"
            frozen_cfg.write_text("data_root: /frozen\nhandoff_root: /frozen\n", encoding="utf-8")

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
                handoff_root=str(handoff),
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
                handoff_root=str(handoff),
                data_root=str(data),
            )
            targets = tmp_path / "targets.csv"
            _write_targets(targets)
            run_dir = runs_root / "run_a"
            logs.materialize_run_inputs(source_cfg, targets, run_dir)

            ctx = resolve_run_context(run_id="run_a", runs_root=str(runs_root))
            self.assertEqual(ctx.run_id, "run_a")
            self.assertEqual(ctx.run_dir, run_dir.resolve())


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
