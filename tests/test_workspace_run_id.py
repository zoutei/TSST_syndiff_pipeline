"""Tests for workspace_run_id and self-contained workspace trees."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_templates_symlink,
    event_templates_symlink_path,
)
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.context import (
    PipelineInvocationContext,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    DIFF_CONFIG_SNAPSHOT_BASENAME,
    SHARED_MASK_FITS_BASENAME,
    workspace_root,
    workspace_tree_name,
)


class TestWorkspaceRunId(unittest.TestCase):
    def test_workspace_tree_name(self):
        self.assertEqual(workspace_tree_name(None), "ws")
        self.assertEqual(workspace_tree_name(""), "ws")
        self.assertEqual(workspace_tree_name("dbg1"), "ws_dbg1")

    def test_debug_workspace_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = os.path.join(tmp, "event")
            cfg = SynDiffConfig(output_dir=event, workspace_run_id="dbg_test")
            ctx = PipelineInvocationContext.from_config(cfg)
            self.assertEqual(
                ctx.workspace_root_path(),
                os.path.join(os.path.abspath(event), "ws_dbg_test"),
            )
            self.assertEqual(
                ctx.workspace("ks_d"),
                os.path.join(os.path.abspath(event), "ws_dbg_test", "ks_d"),
            )
            self.assertEqual(
                ctx.workspace_artifact(SHARED_MASK_FITS_BASENAME),
                os.path.join(
                    os.path.abspath(event), "ws_dbg_test", SHARED_MASK_FITS_BASENAME
                ),
            )

    def test_debug_tree_gets_templates_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp) / "event"
            physical = Path(tmp) / "templates_physical"
            physical.mkdir()
            ensure_event_templates_symlink(event, physical, run_id="dbg_x")
            link = event_templates_symlink_path(event, run_id="dbg_x")
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), physical.resolve())
            self.assertEqual(
                workspace_root(str(event), run_id="dbg_x"),
                str((event / "ws_dbg_x").resolve()),
            )

    def test_config_snapshot_basename(self):
        self.assertEqual(DIFF_CONFIG_SNAPSHOT_BASENAME, "diff_config.yaml")


if __name__ == "__main__":
    unittest.main()
