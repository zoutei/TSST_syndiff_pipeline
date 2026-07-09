"""Tests for star_targets CSV parsing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.star.site_config import (
    load_star_targets,
    star_targets_to_orchestrator_targets,
)


class TestStarTargets(unittest.TestCase):
    def test_load_star_targets_resolves_relative_stars_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site = Path(tmpdir)
            hosts = site / "hosts.csv"
            hosts.write_text("tic_id,gaia_source_id,label\n142748283,,\n", encoding="utf-8")
            targets_path = site / "star_targets.csv"
            targets_path.write_text(
                "sector,camera,ccd,target_name,stars_file,baseline_workspace_run_id,"
                "baseline_diffs,baseline_convolved,phot_bkg,enabled\n"
                f"20,3,2,s20_astrometry,hosts.csv,none,hp_d,hp_c,ks_b_s,true\n",
                encoding="utf-8",
            )
            rows = load_star_targets(targets_path, site_dir=site)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].stars_file, str(hosts.resolve()))
            targets = star_targets_to_orchestrator_targets(rows)
            self.assertEqual(targets[0].label(), "s0020_c3_k2_s20_astrometry")


if __name__ == "__main__":
    unittest.main()
