"""Tests for syndiff_template_* discovery in hotpants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from syndiff_pipeline.difference_imaging.stages import hotpants as hotpants_runner


class TestSyndiffTemplateDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.template_dir = Path(self.tmp.name) / "templates"
        self.template_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name: str) -> Path:
        path = self.template_dir / name
        path.write_bytes(b"fits")
        return path

    def _wcs_table(self, groups: list[tuple[int, float, float]]) -> pd.DataFrame:
        rows = []
        for gid, gdx, gdy in groups:
            rows.append(
                {
                    "group_id": gid,
                    "group_dx": gdx,
                    "group_dy": gdy,
                    "wcs_ok": True,
                }
            )
        return pd.DataFrame(rows)

    def test_duplicate_legacy_and_gz_prefers_gz(self):
        legacy = self._write("syndiff_template_s0015_1_4_dx-0.000_dy-0.000.fits")
        canonical = self._write("syndiff_template_s0015_1_4_dx0.000_dy0.000.fits.gz")
        wcs = self._wcs_table([(5, 0.0, 0.0)])

        out = hotpants_runner.verify_syndiff_templates(
            str(self.template_dir),
            wcs,
            {"x_min": 0, "x_max": 1024, "y_min": 0, "y_max": 1024},
            sector=15,
            camera=1,
            ccd=4,
        )

        self.assertEqual(out[5], str(canonical.resolve()))
        self.assertNotIn(str(legacy.resolve()), out.values())

    def test_missing_group_raises(self):
        self._write("syndiff_template_s0015_1_4_dx0.000_dy0.000.fits.gz")
        wcs = self._wcs_table([(5, 0.0, 0.0), (6, 0.01, 0.0)])

        with self.assertRaises(hotpants_runner.SyndiffTemplateDiscoveryError) as ctx:
            hotpants_runner.verify_syndiff_templates(
                str(self.template_dir),
                wcs,
                {},
                sector=15,
                camera=1,
                ccd=4,
            )
        self.assertIn("Missing syndiff template", str(ctx.exception))
        self.assertIn("group_id=6", str(ctx.exception))

    def test_ensure_propagates_missing_when_syndiff_files_present(self):
        self._write("syndiff_template_s0015_1_4_dx0.000_dy0.000.fits.gz")
        wcs = self._wcs_table([(5, 0.0, 0.0), (6, 0.01, 0.0)])
        cfg = SimpleNamespace(
            template_paths={},
            template_dir=str(self.template_dir),
            sector=15,
            camera=1,
            ccd=4,
        )

        with self.assertRaises(hotpants_runner.SyndiffTemplateDiscoveryError) as ctx:
            hotpants_runner.ensure_template_paths_from_syndiff_or_group_dirs(
                cfg,
                wcs,
                {},
            )
        self.assertIn("Missing syndiff template", str(ctx.exception))
        self.assertNotIn("group_*/ps1_template.fits", str(ctx.exception))

    def test_ensure_falls_back_to_group_dirs_when_no_syndiff_files(self):
        group_dir = self.template_dir / "group_0"
        group_dir.mkdir()
        (group_dir / "ps1_template.fits").write_bytes(b"fits")
        wcs = self._wcs_table([(0, 0.0, 0.0)])
        cfg = SimpleNamespace(
            template_paths={},
            template_dir=str(self.template_dir),
            sector=15,
            camera=1,
            ccd=4,
        )

        hotpants_runner.ensure_template_paths_from_syndiff_or_group_dirs(
            cfg,
            wcs,
            {},
        )
        self.assertEqual(len(cfg.template_paths), 1)
        self.assertIn(0, cfg.template_paths)


if __name__ == "__main__":
    unittest.main()
