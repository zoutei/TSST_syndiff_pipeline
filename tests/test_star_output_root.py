"""Tests for star output path resolution under ``{lane_root}/host_star`` (per-SCC)."""

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

from syndiff_pipeline.common.scc_paths import scc_diff_dir
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


def _fake_ctx(
    *,
    data_root: str,
    sector: int = 20,
    camera: int = 3,
    ccd: int = 2,
    output_store_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=str(data_root),
        sector=sector,
        camera=camera,
        ccd=ccd,
        output_store_name=output_store_name,
    )


class TestStarOutputRoot(unittest.TestCase):
    def test_star_output_root_is_host_star_under_scc_diff_lane(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _fake_ctx(data_root=tmpdir)
            expected = (
                scc_diff_dir(tmpdir, 20, 3, 2, store_name=None) / HOST_STAR_SUBDIR
            )
            self.assertEqual(
                star_output_root(ctx, photometry_run_id="star_full_lc"), expected
            )
            # photometry_run_id no longer namespaces the SCC-lane output.
            self.assertEqual(
                star_output_root(ctx, photometry_run_id=None), expected
            )

    def test_star_output_root_respects_output_store_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _fake_ctx(data_root=tmpdir, output_store_name="alt")
            expected = (
                scc_diff_dir(tmpdir, 20, 3, 2, store_name="alt") / HOST_STAR_SUBDIR
            )
            self.assertEqual(
                star_output_root(ctx, photometry_run_id="run_a"), expected
            )

    def test_resolve_returns_lane_host_star(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _fake_ctx(data_root=tmpdir)
            expected = (
                scc_diff_dir(tmpdir, 20, 3, 2, store_name=None) / HOST_STAR_SUBDIR
            )
            self.assertEqual(
                resolve_star_host_root(ctx, "star_lc_full", photometry_run_id="run_a"),
                expected,
            )

    def test_deprecated_workspace_run_id_warns_and_does_not_affect_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site = Path(tmpdir) / "site"
            site.mkdir()
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

            data_root = Path(tmpdir) / "data"
            ctx = _fake_ctx(data_root=str(data_root))
            expected = (
                scc_diff_dir(str(data_root), 20, 3, 2, store_name=None)
                / HOST_STAR_SUBDIR
            )
            self.assertEqual(
                star_output_root(ctx, photometry_run_id=run_cfg.photometry_run_id),
                expected,
            )


class TestHostStarMasterSkip(unittest.TestCase):
    def test_host_star_label_constant(self):
        from syndiff_pipeline.difference_imaging.support.paths import HOST_STAR_WS_LABEL

        self.assertEqual(HOST_STAR_WS_LABEL, "host_star")


if __name__ == "__main__":
    unittest.main()
