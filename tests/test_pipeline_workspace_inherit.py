"""Tests for workspace_inherit preamble and bootstrap."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import (
    is_workspace_inherit_entry,
    parse_workspace_inherit_spec,
    split_pipeline,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline
from syndiff_pipeline.difference_imaging.support.workspace_inherit import (
    WorkspaceInheritSpec,
    bootstrap_workspace_inherit,
)


class TestPipelineWorkspaceInherit(unittest.TestCase):
    def test_is_workspace_inherit_entry(self):
        self.assertTrue(
            is_workspace_inherit_entry(
                {"workspace_inherit": {"from": "single_hp_kernel", "labels": ["kd_b"]}}
            )
        )
        self.assertFalse(
            is_workspace_inherit_entry(
                {"kind": "hotpants", "workspace_inherit": {"from": "x"}}
            )
        )

    def test_parse_workspace_inherit_spec(self):
        spec = parse_workspace_inherit_spec(
            {
                "workspace_inherit": {
                    "from": "single_hp_kernel",
                    "labels": ["kd_b"],
                    "root_artifacts": ["shared_mask.fits"],
                }
            },
            0,
        )
        self.assertEqual(spec.from_run_id, "single_hp_kernel")
        self.assertEqual(spec.labels, ("kd_b",))
        self.assertEqual(spec.root_artifacts, ("shared_mask.fits",))

    def test_split_pipeline_inherit_and_stage(self):
        pipeline = [
            {
                "workspace_inherit": {
                    "from": "parent",
                    "labels": ["kd_b"],
                }
            },
            {
                "kind": "hotpants",
                "inputs": {"bkg": "kd_b"},
                "output": {"diffs": "mk_d", "convolved": "mk_c"},
            },
        ]
        labels, inherit, stages = split_pipeline(pipeline)
        self.assertEqual(labels, [])
        self.assertEqual(len(inherit), 1)
        self.assertEqual(inherit[0].from_run_id, "parent")
        self.assertEqual(len(stages), 1)

    def test_validate_resume_style_pipeline(self):
        cfg = SynDiffConfig(
            pipeline=[
                {
                    "workspace_inherit": {
                        "from": "single_hp_kernel",
                        "labels": ["kd_b"],
                        "root_artifacts": ["shared_mask.fits"],
                    }
                },
                {
                    "kind": "hotpants",
                    "inputs": {"bkg": "kd_b"},
                    "hp_bgo": 0,
                    "output": {"diffs": "mk_d", "convolved": "mk_c", "bkg": "mk_b"},
                },
                {
                    "kind": "forced_photometry",
                    "inputs": {"diffs": "mk_d"},
                    "psf_type": "prf",
                    "output": "lc_prf_on_mk_diffs",
                },
            ]
        )
        validate_pipeline(cfg)


class TestBootstrapWorkspaceInherit(unittest.TestCase):
    def test_bootstrap_creates_symlinks_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp)
            parent = event / "ws_parent"
            parent.mkdir()
            (parent / "kd_b").mkdir()
            (parent / "kd_b" / "frame.fits").write_bytes(b"fits")
            (parent / "shared_mask.fits").write_bytes(b"mask")

            spec = WorkspaceInheritSpec(
                from_run_id="parent",
                labels=("kd_b",),
                root_artifacts=("shared_mask.fits",),
            )
            bootstrap_workspace_inherit(event, run_id="child", spec=spec)
            child = event / "ws_child"
            self.assertTrue((child / "kd_b").is_symlink())
            self.assertTrue((child / "shared_mask.fits").is_symlink())
            self.assertTrue((child / "kd_b" / "frame.fits").is_file())

            bootstrap_workspace_inherit(event, run_id="child", spec=spec)

    def test_bootstrap_parent_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = WorkspaceInheritSpec(from_run_id="missing", labels=("kd_b",))
            with self.assertRaises(FileNotFoundError):
                bootstrap_workspace_inherit(tmp, run_id="child", spec=spec)

    def test_bootstrap_conflict_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp)
            parent = event / "ws_parent"
            parent.mkdir()
            (parent / "kd_b").mkdir()
            child = event / "ws_child"
            child.mkdir()
            (child / "kd_b").mkdir()

            spec = WorkspaceInheritSpec(from_run_id="parent", labels=("kd_b",))
            with self.assertRaises(RuntimeError):
                bootstrap_workspace_inherit(event, run_id="child", spec=spec)


if __name__ == "__main__":
    unittest.main()
