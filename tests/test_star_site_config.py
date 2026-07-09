"""Tests for star site policy and star_targets merge precedence."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.star.site_config import (
    find_star_target_row,
    load_star_site_policy,
    load_star_targets,
    normalize_ps1_source,
    resolve_star_run_config,
)


class TestStarSiteConfig(unittest.TestCase):
    def test_normalize_ps1_source_legacy_aliases(self):
        self.assertEqual(normalize_ps1_source("zarr", warn_legacy=False), "zarr_download")
        self.assertEqual(normalize_ps1_source("download", warn_legacy=False), "stream")

    def test_merge_precedence_row_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site = Path(tmpdir)
            policy_path = site / "star_config.yaml"
            policy_path.write_text(
                """
defaults:
  cutout_size: 96
  ps1_source: zarr_download
baseline:
  workspace_run_id: none
  diffs: hp_d
  convolved: hp_c
  phot_bkg: ks_b
overrides:
  "20/3/2":
    baseline:
      phot_bkg: ks_b_s
""".strip(),
                encoding="utf-8",
            )
            hosts = site / "hosts.csv"
            hosts.write_text("tic_id,gaia_source_id,label\n1,,\n", encoding="utf-8")
            targets_path = site / "star_targets.csv"
            targets_path.write_text(
                "sector,camera,ccd,target_name,stars_file,baseline_workspace_run_id,"
                "baseline_diffs,baseline_convolved,phot_bkg,enabled\n"
                f"20,3,2,s20_astrometry,{hosts},row_ws,row_hp_d,row_hp_c,row_ks,true\n",
                encoding="utf-8",
            )

            policy = load_star_site_policy(policy_path)
            rows = load_star_targets(targets_path, site_dir=site)
            row = find_star_target_row(rows, "20/3/2")
            run_cfg = resolve_star_run_config(policy, row, site_dir=site)

            self.assertEqual(run_cfg.baseline.workspace_run_id, "row_ws")
            self.assertEqual(run_cfg.baseline.diffs, "row_hp_d")
            self.assertEqual(run_cfg.baseline.convolved, "row_hp_c")
            self.assertEqual(run_cfg.baseline.phot_bkg, "row_ks")
            self.assertEqual(run_cfg.ps1_source, "zarr_download")
            self.assertEqual(run_cfg.stars_file, str(hosts.resolve()))


if __name__ == "__main__":
    unittest.main()
