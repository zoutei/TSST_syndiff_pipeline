"""Tests for the composed SynDiff pipeline spec."""

from __future__ import annotations

import unittest

from syndiff_pipeline.common.orchestration.spec import PipelineSpec
from syndiff_pipeline.pipeline_spec import (
    STAGE_DEPS,
    STAGE_NAMES,
    STAGE_POOL,
    STAGE_SHORT_NAMES,
    SYNDIFF_PIPELINE,
)
from syndiff_pipeline.template_creation.orchestration.stages import TEMPLATE_STAGES


class TestPipelineSpec(unittest.TestCase):
    def test_template_stage_count(self):
        self.assertEqual(len(TEMPLATE_STAGES), 6)

    def test_composed_stage_count_includes_diff(self):
        self.assertEqual(len(STAGE_NAMES), 7)
        self.assertEqual(STAGE_NAMES[-1], "diff")

    def test_template_stage_order(self):
        self.assertEqual(
            tuple(s.name for s in TEMPLATE_STAGES),
            (
                "tess_ffi_download",
                "wcs_grouping",
                "mapping",
                "ps1_download",
                "ps1_process",
                "downsample",
            ),
        )

    def test_downsample_deps(self):
        self.assertEqual(
            STAGE_DEPS["downsample"],
            ["wcs_grouping", "mapping", "ps1_process"],
        )

    def test_diff_depends_on_downsample(self):
        self.assertEqual(STAGE_DEPS["diff"], ["downsample"])

    def test_wcs_grouping_unpooled(self):
        self.assertNotIn("wcs_grouping", STAGE_POOL)

    def test_short_names(self):
        self.assertEqual(STAGE_SHORT_NAMES["mapping"], "map")
        self.assertEqual(STAGE_SHORT_NAMES["ps1_process"], "ps1_pr")

    def test_ps1_process_stream_effective_deps(self):
        from syndiff_pipeline.template_creation.orchestration.stage_params import (
            parse_stage_params,
        )

        stream_stages = parse_stage_params({"ps1_process": {"ps1_source": "stream"}})
        self.assertEqual(
            SYNDIFF_PIPELINE.effective_stage_deps("ps1_process", stream_stages),
            ["mapping"],
        )
        zarr_stages = parse_stage_params({"ps1_process": {"ps1_source": "zarr"}})
        self.assertEqual(
            SYNDIFF_PIPELINE.effective_stage_deps("ps1_process", zarr_stages),
            ["ps1_download"],
        )

    def test_upstream_closure_for_partial_run(self):
        closure = SYNDIFF_PIPELINE.run_stage_closure(["downsample"])
        self.assertEqual(
            closure,
            {
                "downsample",
                "wcs_grouping",
                "tess_ffi_download",
                "mapping",
                "ps1_process",
                "ps1_download",
            },
        )

    def test_diff_only_artifact_verify_closure(self):
        from syndiff_pipeline.common.orchestration.spec import DIFF_VERIFY_UPSTREAM

        closure = SYNDIFF_PIPELINE.artifact_verify_closure(["diff"])
        self.assertEqual(closure, frozenset({"diff"}) | DIFF_VERIFY_UPSTREAM)
        self.assertNotIn("mapping", closure)
        self.assertNotIn("ps1_download", closure)
        self.assertNotIn("ps1_process", closure)

    def test_non_diff_run_uses_full_closure_for_verify(self):
        closure = SYNDIFF_PIPELINE.artifact_verify_closure(["downsample"])
        self.assertEqual(
            closure,
            SYNDIFF_PIPELINE.run_stage_closure(["downsample"]),
        )

    def test_downstream_from_mapping(self):
        downstream = SYNDIFF_PIPELINE.downstream_stages("mapping")
        self.assertEqual(
            downstream,
            ["ps1_download", "ps1_process", "downsample", "diff"],
        )

    def test_stages_in_pool(self):
        self.assertEqual(
            SYNDIFF_PIPELINE.stages_in_pool("network"),
            ["tess_ffi_download", "ps1_download"],
        )

    def test_unknown_stage_raises(self):
        with self.assertRaises(KeyError):
            SYNDIFF_PIPELINE.require("not_a_stage")

    def test_resolve_stage_name_full(self):
        self.assertEqual(SYNDIFF_PIPELINE.resolve_stage_name("mapping"), "mapping")

    def test_resolve_stage_name_short(self):
        self.assertEqual(SYNDIFF_PIPELINE.resolve_stage_name("map"), "mapping")
        self.assertEqual(SYNDIFF_PIPELINE.resolve_stage_name("ps1_pr"), "ps1_process")

    def test_resolve_stage_name_unknown(self):
        with self.assertRaises(ValueError) as ctx:
            SYNDIFF_PIPELINE.resolve_stage_name("not_a_stage")
        self.assertIn("Unknown stage", str(ctx.exception))

    def test_resolve_stage_name_empty(self):
        with self.assertRaises(ValueError):
            SYNDIFF_PIPELINE.resolve_stage_name("  ")

    def test_direct_dependents(self):
        self.assertEqual(
            SYNDIFF_PIPELINE.direct_dependents("mapping"),
            ["ps1_download", "downsample"],
        )

    def test_pipeline_spec_view(self):
        other = PipelineSpec(name="other", stages=TEMPLATE_STAGES)
        self.assertEqual(other.stage_names, tuple(s.name for s in TEMPLATE_STAGES))
