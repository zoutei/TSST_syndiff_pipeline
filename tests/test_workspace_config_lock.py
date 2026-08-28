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
    versioned_fingerprint,
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

    def test_fingerprint_ignores_epsf_and_centroids(self):
        base = [
            {"kind": "shared_mask"},
            {
                "kind": "hotpants",
                "inputs": {"bkg": "ks_b"},
                "output": {"diffs": "hp_d", "bkg": "hp_b"},
            },
        ]
        extended = base + [
            {"kind": "epsf", "inputs": {"diffs": "hp_d"}, "output": "epsf_r1"},
            {
                "kind": "centroids",
                "inputs": {"diffs": "hp_d", "epsf": "epsf_r1"},
                "output": "centroids_r1",
            },
            {
                "kind": "per_ffi_wcs",
                "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
                "output": "wcs",
            },
        ]
        a = _minimal_cfg(pipeline=base)
        b = _minimal_cfg(pipeline=extended)
        self.assertEqual(diff_config_fingerprint(a), diff_config_fingerprint(b))

    def test_fingerprint_epsf_param_change_still_ignored(self):
        base = _minimal_cfg(
            pipeline=[
                {"kind": "shared_mask"},
                {
                    "kind": "epsf",
                    "inputs": {"diffs": "hp_d"},
                    "output": "epsf_r1",
                    "tile_nx": 5,
                },
            ]
        )
        other = _minimal_cfg(
            pipeline=[
                {"kind": "shared_mask"},
                {
                    "kind": "epsf",
                    "inputs": {"diffs": "hp_d"},
                    "output": "epsf_r1",
                    "tile_nx": 3,
                },
            ]
        )
        self.assertEqual(diff_config_fingerprint(base), diff_config_fingerprint(other))

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
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / DIFF_CONFIG_SNAPSHOT_BASENAME).write_text("snap\n")
            (ws / "diff_config.fingerprint").write_text(versioned_fingerprint(cfg) + "\n")
            assert_workspace_config_lock(ws, cfg)


# ── Execution-resource params must never participate in the lock ─────────────


def _cfg_with_resources(n_jobs: int, os_n_jobs: int = 4) -> SynDiffConfig:
    """Same recipe, differing only in execution-resource knobs."""
    return _minimal_cfg(
        pipeline=[
            {"kind": "shared_mask"},
            {
                "kind": "background_estimate",
                "inputs": {"convolved": "tmpl_conv"},
                "output": {"diffs": "ks_d", "phot_bkg": "ks_b"},
                "background_estimate_n_jobs": n_jobs,
            },
            {
                "kind": "hotpants",
                "inputs": {"bkg": "ks_b"},
                "output": {"diffs": "hp_d"},
                "hotpants_n_jobs": n_jobs,
                "hotpants_os_n_jobs": os_n_jobs,
                "hp_ko": 2,
            },
        ]
    )


class TestResourceParamsExcludedFromLock(unittest.TestCase):
    def test_legacy_flat_resource_keys_do_not_move_fingerprint(self):
        """The bug this whole change exists to fix: 48 -> 64 must not relock."""
        a = _cfg_with_resources(48)
        b = _cfg_with_resources(64)
        self.assertEqual(
            diff_config_fingerprint(a, version=2),
            diff_config_fingerprint(b, version=2),
        )

    def test_nested_resources_block_does_not_move_fingerprint(self):
        a = _minimal_cfg(
            pipeline=[{"kind": "hotpants", "hp_ko": 2, "resources": {"n_jobs": 8}}]
        )
        b = _minimal_cfg(
            pipeline=[{"kind": "hotpants", "hp_ko": 2, "resources": {"n_jobs": 64}}]
        )
        self.assertEqual(
            diff_config_fingerprint(a, version=2),
            diff_config_fingerprint(b, version=2),
        )

    def test_nested_and_flat_spellings_agree(self):
        """A migrated config must hash identically to its unmigrated form."""
        flat = _minimal_cfg(
            pipeline=[{"kind": "hotpants", "hp_ko": 2, "hotpants_n_jobs": 48}]
        )
        nested = _minimal_cfg(
            pipeline=[{"kind": "hotpants", "hp_ko": 2, "resources": {"n_jobs": 64}}]
        )
        self.assertEqual(
            diff_config_fingerprint(flat, version=2),
            diff_config_fingerprint(nested, version=2),
        )

    def test_background_step_n_jobs_stripped(self):
        a = _minimal_cfg(
            pipeline=[
                {
                    "kind": "background_temporal_smoothing",
                    "steps": {"spatial": {"box_size": 64, "n_jobs": 2}},
                }
            ]
        )
        b = _minimal_cfg(
            pipeline=[
                {
                    "kind": "background_temporal_smoothing",
                    "steps": {"spatial": {"box_size": 64, "n_jobs": 32}},
                }
            ]
        )
        self.assertEqual(
            diff_config_fingerprint(a, version=2),
            diff_config_fingerprint(b, version=2),
        )

    def test_real_recipe_change_still_moves_fingerprint(self):
        """The guard must still fire for anything that changes output bytes."""
        a = _cfg_with_resources(48)
        b = _minimal_cfg(
            pipeline=[
                dict(s, hp_ko=9) if s.get("kind") == "hotpants" else s
                for s in _cfg_with_resources(48).pipeline
            ]
        )
        self.assertNotEqual(
            diff_config_fingerprint(a, version=2),
            diff_config_fingerprint(b, version=2),
        )

    def test_denylisted_knobs_still_hashed(self):
        """use_patch_cache is NOT bit-exact; it must stay part of identity."""
        a = _minimal_cfg(
            pipeline=[{"kind": "convolved_templates", "use_patch_cache": False}]
        )
        b = _minimal_cfg(
            pipeline=[{"kind": "convolved_templates", "use_patch_cache": True}]
        )
        self.assertNotEqual(
            diff_config_fingerprint(a, version=2),
            diff_config_fingerprint(b, version=2),
        )


class TestFingerprintMigration(unittest.TestCase):
    def test_legacy_v1_fingerprint_self_heals_to_v2(self):
        cfg = _cfg_with_resources(48)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / DIFF_CONFIG_SNAPSHOT_BASENAME).write_text("snap\n")
            fp_path = ws / "diff_config.fingerprint"
            fp_path.write_text(diff_config_fingerprint(cfg, version=1) + "\n")

            assert_workspace_config_lock(ws, cfg)

            migrated = fp_path.read_text().strip()
            self.assertTrue(migrated.startswith("v2:"))
            self.assertEqual(migrated, versioned_fingerprint(cfg))

    def test_migrated_lane_then_accepts_a_resource_retune(self):
        """End-to-end of the motivating scenario, at the lock level."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / DIFF_CONFIG_SNAPSHOT_BASENAME).write_text("snap\n")
            fp_path = ws / "diff_config.fingerprint"
            # Lane was frozen under the old rules at n_jobs=48 ...
            fp_path.write_text(
                diff_config_fingerprint(_cfg_with_resources(48), version=1) + "\n"
            )
            # ... first touch migrates it ...
            assert_workspace_config_lock(ws, _cfg_with_resources(48))
            # ... and the retune to 64 no longer raises.
            assert_workspace_config_lock(ws, _cfg_with_resources(64))

    def test_legacy_v1_with_genuinely_changed_recipe_still_raises(self):
        cfg = _cfg_with_resources(48)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / DIFF_CONFIG_SNAPSHOT_BASENAME).write_text("snap\n")
            (ws / "diff_config.fingerprint").write_text(
                diff_config_fingerprint(cfg, version=1) + "\n"
            )
            changed = _minimal_cfg(pipeline=[{"kind": "shared_mask", "strapsize": 99}])
            with self.assertRaises(WorkspaceConfigMismatchError):
                assert_workspace_config_lock(ws, changed)

    def test_v2_mismatch_still_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / DIFF_CONFIG_SNAPSHOT_BASENAME).write_text("snap\n")
            (ws / "diff_config.fingerprint").write_text(
                versioned_fingerprint(_cfg_with_resources(48)) + "\n"
            )
            changed = _minimal_cfg(pipeline=[{"kind": "shared_mask", "strapsize": 99}])
            with self.assertRaises(WorkspaceConfigMismatchError):
                assert_workspace_config_lock(ws, changed)

    def test_snapshot_writer_emits_versioned_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(output_dir=tmp, workspace_run_id="lock_test")
            ctx = PipelineInvocationContext.from_config(cfg)
            write_immutable_workspace_config_snapshot(ctx, cfg)
            fp = Path(ctx.cfg.output_dir) / "diff_config.fingerprint"
            self.assertEqual(fp.read_text().strip(), versioned_fingerprint(cfg))

    def test_write_snapshot_once_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(output_dir=tmp, workspace_run_id="lock_test")
            ctx = PipelineInvocationContext.from_config(cfg)
            write_immutable_workspace_config_snapshot(ctx, cfg)
            # Lock artifacts live at the lane root (cfg.output_dir) directly
            # (wave A-3 removed the ws[_{run_id}]/ tree).
            snap = Path(ctx.cfg.output_dir) / DIFF_CONFIG_SNAPSHOT_BASENAME
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
