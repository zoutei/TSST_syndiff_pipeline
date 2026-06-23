"""Tests for pipeline external_workspaces preamble entries."""

from __future__ import annotations

import unittest

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import (
    is_external_workspaces_entry,
    split_pipeline,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline

_FORCED_PHOT = {
    "kind": "forced_photometry",
    "inputs": {"diffs": "hp_d"},
    "psf_type": "prf",
    "output": "lc_prf_on_diffs",
}


class TestPipelineExternalWorkspaces(unittest.TestCase):
    def test_is_external_workspaces_entry(self):
        self.assertTrue(
            is_external_workspaces_entry({"external_workspaces": ["hp_d"]})
        )
        self.assertFalse(
            is_external_workspaces_entry(
                {"kind": "forced_photometry", "external_workspaces": ["hp_d"]}
            )
        )
        self.assertFalse(is_external_workspaces_entry({"kind": "shared_mask"}))
        self.assertFalse(is_external_workspaces_entry("not a dict"))

    def test_split_pipeline_preamble_and_stage(self):
        pipeline = [
            {"external_workspaces": ["hp_d", "hp_b"]},
            _FORCED_PHOT,
        ]
        labels, inherit, stages = split_pipeline(pipeline)
        self.assertEqual(labels, ["hp_d", "hp_b"])
        self.assertEqual(inherit, [])
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0][0], 1)
        self.assertEqual(stages[0][1]["kind"], "forced_photometry")

    def test_split_pipeline_multiple_preambles(self):
        pipeline = [
            {"external_workspaces": ["hp_d"]},
            {"external_workspaces": ["hp_b"]},
            _FORCED_PHOT,
        ]
        labels, inherit, stages = split_pipeline(pipeline)
        self.assertEqual(labels, ["hp_d", "hp_b"])
        self.assertEqual(inherit, [])
        self.assertEqual(len(stages), 1)

    def test_split_pipeline_preamble_after_stage_fails(self):
        pipeline = [
            _FORCED_PHOT,
            {"external_workspaces": ["hp_d"]},
        ]
        with self.assertRaises(ValueError) as ctx:
            split_pipeline(pipeline)
        self.assertIn("before the first", str(ctx.exception))

    def test_split_pipeline_kind_and_external_fails(self):
        pipeline = [
            {"kind": "forced_photometry", "external_workspaces": ["hp_d"]},
        ]
        with self.assertRaises(ValueError) as ctx:
            split_pipeline(pipeline)
        self.assertIn("cannot set both", str(ctx.exception))

    def test_split_pipeline_extra_keys_on_preamble_fails(self):
        pipeline = [{"external_workspaces": ["hp_d"], "note": "resume"}]
        with self.assertRaises(ValueError) as ctx:
            split_pipeline(pipeline)
        self.assertIn("unexpected key", str(ctx.exception))

    def test_validate_preamble_plus_forced_photometry(self):
        cfg = SynDiffConfig(
            pipeline=[
                {"external_workspaces": ["hp_d"]},
                dict(_FORCED_PHOT),
            ]
        )
        validate_pipeline(cfg)

    def test_validate_forced_photometry_without_source_fails(self):
        cfg = SynDiffConfig(pipeline=[dict(_FORCED_PHOT)])
        with self.assertRaises(ValueError) as ctx:
            validate_pipeline(cfg)
        self.assertIn("hp_d", str(ctx.exception))
        self.assertIn("not produced", str(ctx.exception))

    def test_validate_legacy_pipeline_external_workspace_labels(self):
        cfg = SynDiffConfig(
            pipeline=[dict(_FORCED_PHOT)],
            pipeline_external_workspace_labels=["hp_d"],
        )
        validate_pipeline(cfg)

    def test_validate_preamble_only_fails(self):
        cfg = SynDiffConfig(pipeline=[{"external_workspaces": ["hp_d"]}])
        with self.assertRaises(ValueError) as ctx:
            validate_pipeline(cfg)
        self.assertIn("no executable stages", str(ctx.exception))

    def test_validate_preamble_and_legacy_merged(self):
        cfg = SynDiffConfig(
            pipeline=[
                {"external_workspaces": ["hp_d"]},
                {
                    "kind": "forced_photometry",
                    "inputs": {"diffs": "bkg_rough"},
                    "psf_type": "prf",
                    "output": "lc_other",
                },
            ],
            pipeline_external_workspace_labels=["bkg_rough"],
        )
        validate_pipeline(cfg)


if __name__ == "__main__":
    unittest.main()
