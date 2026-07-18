"""Tests for star oversampling_factor config wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.scc_paths import (
    scc_mapping_dir,
    scc_mapping_master_pixels2skycells,
    scc_mapping_master_skycells_csv,
    scc_templates_dir,
)
from syndiff_pipeline.star.context import _mapping_paths
from syndiff_pipeline.star.site_config import (
    StarBaselineConfig,
    StarSitePolicy,
    StarTargetRow,
    resolve_star_run_config,
)


def _target() -> Target:
    return Target(
        sector=20,
        camera=3,
        ccd=3,
        target_ra=1.0,
        target_dec=2.0,
        target_name="t1",
    )


class TestStarOversamplingConfig(unittest.TestCase):
    def test_defaults_oversampling_factor(self):
        policy = StarSitePolicy(
            deployment_file="deployment.yaml",
            defaults={"oversampling_factor": 2, "cutout_size": 48},
            baseline=StarBaselineConfig(),
            photometry={"methods": []},
            epsf={},
            overrides={},
        )
        row = StarTargetRow(target=_target(), stars_file="stars.csv")
        run = resolve_star_run_config(policy, row, site_dir=tempfile.mkdtemp())
        self.assertEqual(run.oversampling_factor, 2)

    def test_mapping_paths_use_scc_oversampling(self):
        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            expected_csv = scc_mapping_master_skycells_csv(
                data_root, 20, 3, 3, oversampling_factor=2
            )
            expected_fits = scc_mapping_master_pixels2skycells(
                data_root, 20, 3, 3, oversampling_factor=2
            )
            expected_csv.parent.mkdir(parents=True, exist_ok=True)
            expected_csv.write_text("NAME\n", encoding="utf-8")
            expected_fits.write_bytes(b"")

            mapping_dir, mapping_csv, master_fits = _mapping_paths(
                str(data_root), target, oversampling_factor=2
            )
            self.assertEqual(
                Path(mapping_dir),
                scc_mapping_dir(data_root, 20, 3, 3, oversampling_factor=2),
            )
            self.assertEqual(Path(mapping_csv), expected_csv)
            self.assertEqual(Path(master_fits), expected_fits)
            self.assertEqual(
                scc_templates_dir(data_root, 20, 3, 3, oversampling_factor=2),
                data_root / "s0020" / "c3" / "k3" / "templates" / "oversampling_2",
            )


if __name__ == "__main__":
    unittest.main()
