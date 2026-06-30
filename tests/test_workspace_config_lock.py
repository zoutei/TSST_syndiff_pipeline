"""Tests for workspace config fingerprint lock and immutable snapshot."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.context import (
    PipelineInvocationContext,
)
from syndiff_pipeline.difference_imaging.orchestration.workspace_lock import (
    WorkspaceConfigMismatchError,
    assert_workspace_config_lock,
    diff_config_fingerprint,
    write_immutable_workspace_config_snapshot,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    DIFF_CONFIG_SNAPSHOT_BASENAME,
)


def _minimal_cfg(**kwargs) -> SynDiffConfig:
    base = dict(
        output_dir="/tmp/event",
        pipeline=[
            {"kind": "shared_mask"},
            {
                "kind": "forced_photometry",
                "inputs": {"diffs": "hp_d"},
                "output": "lc",
                "methods": [
                    {"name": "prf", "type": "psf", "psf_type": "prf"},
                ],
            },
        ],
        sector=20,
        camera=3,
        ccd=3,
        workspace_run_id="test_run",
    )
    base.update(kwargs)
    return SynDiffConfig(**base)


class TestWorkspaceConfigLock(unittest.TestCase):
    def test_fingerprint_stable(self):
        cfg = _minimal_cfg()
        self.assertEqual(diff_config_fingerprint(cfg), diff_config_fingerprint(cfg))

    def test_fingerprint_changes_with_pipeline(self):
        a = _minimal_cfg()
        b = _minimal_cfg(pipeline=[{"kind": "shared_mask"}])
        self.assertNotEqual(diff_config_fingerprint(a), diff_config_fingerprint(b))

    def test_assert_lock_no_snapshot_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert_workspace_config_lock(tmp, _minimal_cfg())

    def test_assert_lock_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / DIFF_CONFIG_SNAPSHOT_BASENAME).write_text("old\n")
            (ws / "diff_config.fingerprint").write_text("deadbeefdeadbeef\n")
            with self.assertRaises(WorkspaceConfigMismatchError):
                assert_workspace_config_lock(ws, _minimal_cfg())

    def test_assert_lock_match_ok(self):
        cfg = _minimal_cfg()
        fp = diff_config_fingerprint(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / DIFF_CONFIG_SNAPSHOT_BASENAME).write_text("snap\n")
            (ws / "diff_config.fingerprint").write_text(fp + "\n")
            assert_workspace_config_lock(ws, cfg)

    def test_write_snapshot_once_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(output_dir=tmp, workspace_run_id="lock_test")
            ctx = PipelineInvocationContext.from_config(cfg)
            write_immutable_workspace_config_snapshot(ctx, cfg)
            snap = Path(ctx.workspace_artifact(DIFF_CONFIG_SNAPSHOT_BASENAME))
            fp = snap.parent / "diff_config.fingerprint"
            self.assertTrue(snap.is_file())
            self.assertTrue(fp.is_file())
            mode = stat.S_IMODE(snap.stat().st_mode)
            self.assertEqual(mode, 0o444)

            write_immutable_workspace_config_snapshot(ctx, cfg)

    def test_write_snapshot_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(output_dir=tmp, workspace_run_id="lock_test")
            ctx = PipelineInvocationContext.from_config(cfg)
            write_immutable_workspace_config_snapshot(ctx, cfg)
            other = _minimal_cfg(
                output_dir=tmp,
                workspace_run_id="lock_test",
                pipeline=[{"kind": "shared_mask"}],
            )
            with self.assertRaises(WorkspaceConfigMismatchError):
                write_immutable_workspace_config_snapshot(ctx, other)


if __name__ == "__main__":
    unittest.main()
