"""Tests for star output path resolution under phot_{run_id}/host_star."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.support.paths import photometry_root
from syndiff_pipeline.star.runner import (
    HOST_STAR_SUBDIR,
    resolve_star_host_root,
    star_output_root,
)
from syndiff_pipeline.star.site_config import (
    find_star_target_row,
    load_star_site_policy,
    load_star_targets,
    resolve_star_run_config,
)


def _fake_ctx(*, event_dir: Path, baseline_ws: Path) -> SimpleNamespace:
    return SimpleNamespace(
        event_dir=str(event_dir),
        baseline_workspace_dir=str(baseline_ws),
    )


class TestStarOutputRoot(unittest.TestCase):
    def test_star_output_root_is_host_star_under_photometry_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            event = Path(tmpdir) / "events" / "s0020_c3_k2_s20_astrometry"
            baseline = event / "ws_star_full_lc"
            baseline.mkdir(parents=True)
            ctx = _fake_ctx(event_dir=event, baseline_ws=baseline)
            self.assertEqual(
                star_output_root(ctx, photometry_run_id="star_full_lc"),
                Path(photometry_root(str(event), "star_full_lc")) / HOST_STAR_SUBDIR,
            )
            self.assertEqual(
                star_output_root(ctx, photometry_run_id=None),
                Path(photometry_root(str(event), None)) / HOST_STAR_SUBDIR,
            )

    def test_resolve_returns_phot_host_star(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            event = Path(tmpdir) / "evt"
            baseline = event / "ws_star_full_lc"
            host = Path(photometry_root(str(event), "run_a")) / HOST_STAR_SUBDIR
            host.mkdir(parents=True)
            ctx = _fake_ctx(event_dir=event, baseline_ws=baseline)
            self.assertEqual(
                resolve_star_host_root(ctx, "star_lc_full", photometry_run_id="run_a"),
                host,
            )

    def test_deprecated_workspace_run_id_warns_and_does_not_affect_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site = Path(tmpdir)
            policy_path = site / "star_config.yaml"
            policy_path.write_text(
                """
defaults:
  workspace_run_id: star_lc_full
  photometry_run_id: star_full_lc
baseline:
  workspace_run_id: star_full_lc
  diffs: hp_d
""".strip(),
                encoding="utf-8",
            )
            hosts = site / "hosts.csv"
            hosts.write_text("tic_id,gaia_source_id,label\n1,,\n", encoding="utf-8")
            targets_path = site / "star_targets.csv"
            targets_path.write_text(
                "sector,camera,ccd,target_name,stars_file,baseline_workspace_run_id,"
                "baseline_diffs,baseline_convolved,phot_bkg,enabled\n"
                f"20,3,2,s20_astrometry,{hosts},,,,,true\n",
                encoding="utf-8",
            )
            policy = load_star_site_policy(policy_path)
            rows = load_star_targets(targets_path, site_dir=site)
            row = find_star_target_row(rows, "s0020_c3_k2")
            with self.assertLogs(
                "syndiff_pipeline.star.site_config", level=logging.WARNING
            ) as cm:
                run_cfg = resolve_star_run_config(policy, row, site_dir=site)
            self.assertTrue(any("deprecated" in m for m in cm.output))
            self.assertEqual(run_cfg.workspace_run_id, "star_lc_full")
            self.assertEqual(run_cfg.photometry_run_id, "star_full_lc")
            self.assertEqual(run_cfg.baseline.workspace_run_id, "star_full_lc")

            event = site / "events" / "s0020_c3_k2_s20_astrometry"
            baseline = event / "ws_star_full_lc"
            baseline.mkdir(parents=True)
            ctx = _fake_ctx(event_dir=event, baseline_ws=baseline)
            expected = Path(photometry_root(str(event), "star_full_lc")) / "host_star"
            self.assertEqual(
                star_output_root(ctx, photometry_run_id=run_cfg.photometry_run_id),
                expected,
            )
            self.assertNotEqual(
                star_output_root(ctx, photometry_run_id=run_cfg.photometry_run_id),
                event / "star_star_lc_full",
            )


class TestHostStarMasterSkip(unittest.TestCase):
    def test_host_star_label_constant(self):
        from syndiff_pipeline.difference_imaging.support.paths import HOST_STAR_WS_LABEL

        self.assertEqual(HOST_STAR_WS_LABEL, "host_star")


if __name__ == "__main__":
    unittest.main()
