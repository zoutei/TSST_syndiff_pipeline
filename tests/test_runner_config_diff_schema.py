"""Tests for RunnerConfig's unified (schema v2) ``diff:`` block support.

Covers the additive-only contract for this wave: v1 (``diff_config:``
pointer) keeps working byte-identically, v2 (embedded ``diff:``) is new, and
exactly one of the two may be used per config. See CONTRACT.md and
CLAUDE.md's site-config table for the schema this exercises.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from syndiff_pipeline.template_creation.orchestration.runner_config import (
    RunnerConfig,
    load_and_materialize_runner_config,
    load_runner_config,
    runner_config_to_dict,
    write_runner_config,
)
from tests.site_fixtures import write_site_config, write_unified_site_config

_MINIMAL_DIFF_FLAT: dict = {
    "defaults": {"n_jobs": 4},
    "paths": {"output_store_name": "l4_split_smoke"},
    "pipeline": [{"kind": "shared_mask"}, {"kind": "hotpants"}],
    "condor": {"request_cpus": 4, "request_memory": 32000, "host_stats_min_mem_mb": 32000},
}

_MINIMAL_DIFF_NESTED_CONDOR: dict = {
    "defaults": {"n_jobs": 4},
    "paths": {"output_store_name": "l4_split_smoke"},
    "pipeline": [{"kind": "shared_mask"}, {"kind": "hotpants"}],
    "condor": {
        "diff_prep": {"request_cpus": 4, "request_memory": 8000},
        "background_estimate": {"request_cpus": 32, "request_memory": 200000},
        "diff": {"request_cpus": 8, "request_memory": 16000},
    },
}


class _TempSiteMixin:
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.site = self.root / "site"
        self.handoff = self.root / "handoff"
        self.data = self.root / "data"
        self.handoff.mkdir(parents=True)
        self.data.mkdir(parents=True)


class TestUnifiedSchemaParsing(_TempSiteMixin, unittest.TestCase):
    def test_v2_parses(self):
        write_unified_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
            diff=_MINIMAL_DIFF_FLAT,
        )
        cfg = load_runner_config(self.site / "pipeline.yaml")
        self.assertEqual(cfg.diff_config_path, "")
        self.assertIsNotNone(cfg.diff)
        self.assertEqual(len(cfg.diff.pipeline), 2)
        self.assertEqual(cfg.diff.defaults["n_jobs"], 4)
        self.assertEqual(cfg.diff.source_dir, str(self.site.resolve()))

    def test_diff_optional(self):
        """Template-only / photometry-only configs legitimately have no diff: block."""
        write_unified_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
            diff=None,
        )
        cfg = load_runner_config(self.site / "pipeline.yaml")
        self.assertIsNone(cfg.diff)
        self.assertEqual(cfg.diff_config_path, "")

    def test_v1_pointer_still_works_unaffected(self):
        """Byte-identical-behaviour guarantee: the old diff_config: pointer form."""
        diff_yaml = self.site / "diff_config.yaml"
        diff_yaml.parent.mkdir(parents=True, exist_ok=True)
        diff_yaml.write_text(
            yaml.safe_dump(_MINIMAL_DIFF_FLAT, sort_keys=False), encoding="utf-8"
        )
        write_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
        )
        text = (self.site / "pipeline.yaml").read_text(encoding="utf-8")
        text = text.replace(
            "deployment_file: deployment.yaml",
            "deployment_file: deployment.yaml\ndiff_config: diff_config.yaml",
        )
        (self.site / "pipeline.yaml").write_text(text, encoding="utf-8")

        cfg = load_runner_config(self.site / "pipeline.yaml")
        self.assertIsNone(cfg.diff)
        self.assertTrue(cfg.diff_config_path.endswith("diff_config.yaml"))

    def test_both_forms_rejected(self):
        write_unified_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
            diff=_MINIMAL_DIFF_FLAT,
            extra={"diff_config": "diff_config.yaml"},
        )
        with self.assertRaises(ValueError) as ctx:
            load_runner_config(self.site / "pipeline.yaml")
        msg = str(ctx.exception)
        self.assertIn("diff", msg)
        self.assertIn("exactly one", msg)

    def test_event_kinds_rejected(self):
        for kind in ("astrometry", "forced_photometry", "photometry"):
            with self.subTest(kind=kind):
                bad_diff = dict(_MINIMAL_DIFF_FLAT)
                bad_diff["pipeline"] = [{"kind": "shared_mask"}, {"kind": kind}]
                write_unified_site_config(
                    self.site / "pipeline.yaml",
                    workspace_root=str(self.handoff),
                    data_root=str(self.data),
                    diff=bad_diff,
                )
                with self.assertRaises(ValueError) as ctx:
                    load_runner_config(self.site / "pipeline.yaml")
                self.assertIn(kind, str(ctx.exception))

    def test_dead_keys_rejected(self):
        for dead_key, val in (
            ("additional_forced_targets", [{"target_name": "x"}]),
            ("per_event_force_targets", {"2020ut": [{"target_name": "x"}]}),
        ):
            with self.subTest(dead_key=dead_key):
                bad_diff = dict(_MINIMAL_DIFF_FLAT)
                bad_diff[dead_key] = val
                write_unified_site_config(
                    self.site / "pipeline.yaml",
                    workspace_root=str(self.handoff),
                    data_root=str(self.data),
                    diff=bad_diff,
                )
                with self.assertRaises(ValueError) as ctx:
                    load_runner_config(self.site / "pipeline.yaml")
                self.assertIn(dead_key, str(ctx.exception))


class TestVerbatimFreeze(_TempSiteMixin, unittest.TestCase):
    def _load(self, diff: dict) -> RunnerConfig:
        write_unified_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
            diff=diff,
        )
        return load_runner_config(self.site / "pipeline.yaml")

    def test_flat_condor_survives_verbatim_freeze(self):
        cfg = self._load(_MINIMAL_DIFF_FLAT)
        frozen = runner_config_to_dict(cfg)
        # Verbatim: the frozen diff.condor is still the flat mapping, NOT
        # normalized into condor_by_stage's diff_prep/background_estimate/diff
        # nested shape -- that normalization only happens in the in-memory
        # DiffSitePolicy.condor_by_stage, never in what's written to disk.
        self.assertEqual(frozen["diff"]["condor"], _MINIMAL_DIFF_FLAT["condor"])
        self.assertNotIn("diff_prep", frozen["diff"]["condor"])
        self.assertIn("source_dir", frozen["diff"])

    def test_nested_condor_survives_verbatim_freeze(self):
        cfg = self._load(_MINIMAL_DIFF_NESTED_CONDOR)
        frozen = runner_config_to_dict(cfg)
        self.assertEqual(frozen["diff"]["condor"], _MINIMAL_DIFF_NESTED_CONDOR["condor"])

    def test_no_diff_key_when_diff_none(self):
        write_unified_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
            diff=None,
        )
        cfg = load_runner_config(self.site / "pipeline.yaml")
        frozen = runner_config_to_dict(cfg)
        self.assertNotIn("diff", frozen)


class TestFrozenRoundTrip(_TempSiteMixin, unittest.TestCase):
    def _write_and_load(self, diff: dict) -> tuple[RunnerConfig, Path]:
        write_unified_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
            diff=diff,
        )
        cfg = load_runner_config(self.site / "pipeline.yaml")
        run_dir = self.root / "runs" / "20260827_000000"
        run_dir.mkdir(parents=True)
        frozen_path = run_dir / "config.yaml"
        write_runner_config(cfg, frozen_path)
        return cfg, frozen_path

    def test_frozen_to_loaded_round_trip_is_lossless(self):
        cfg, frozen_path = self._write_and_load(_MINIMAL_DIFF_NESTED_CONDOR)
        reloaded = load_and_materialize_runner_config(frozen_path)
        self.assertIsNotNone(reloaded.diff)
        self.assertEqual(reloaded.diff.pipeline, cfg.diff.pipeline)
        self.assertEqual(reloaded.diff.defaults, cfg.diff.defaults)
        self.assertEqual(reloaded.diff.paths, cfg.diff.paths)
        self.assertEqual(reloaded.diff.condor_by_stage.keys(), cfg.diff.condor_by_stage.keys())
        for stage in cfg.diff.condor_by_stage:
            self.assertEqual(
                reloaded.diff.condor_by_stage[stage], cfg.diff.condor_by_stage[stage]
            )
        # source_dir survives the round trip as the ORIGINAL site dir, not the
        # runs/{run_id}/ directory the frozen file now lives in.
        self.assertEqual(reloaded.diff.source_dir, str(self.site.resolve()))
        self.assertNotEqual(reloaded.diff.source_dir, str(frozen_path.parent.resolve()))
        # Freezing the reloaded config again reproduces the same diff: block.
        frozen_again = runner_config_to_dict(reloaded)
        frozen_first = runner_config_to_dict(cfg)
        self.assertEqual(frozen_again["diff"], frozen_first["diff"])

    def test_hand_edited_frozen_condor_is_picked_up_on_reload(self):
        _cfg, frozen_path = self._write_and_load(_MINIMAL_DIFF_NESTED_CONDOR)

        with frozen_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw["diff"]["condor"]["background_estimate"]["request_memory"] = 999999
        with frozen_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False)

        reloaded = load_and_materialize_runner_config(frozen_path)
        self.assertEqual(
            reloaded.diff.condor_by_stage["background_estimate"].request_memory, 999999
        )
        # The hand-edit is a live retune target -- it must not be
        # second-guessed/normalized by a round trip through parsing.
        self.assertEqual(
            reloaded.diff.raw["condor"]["background_estimate"]["request_memory"], 999999
        )


if __name__ == "__main__":
    unittest.main()
