"""Tests for diff template handoff bootstrap and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from astropy.io import fits
import numpy as np

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.execute import _load_template_handoff
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline


def _minimal_handoff(event_dir: Path, ref_fits: Path) -> None:
    event_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "schema_version": 1,
        "reference_ffi_path": str(ref_fits),
        "reference_ffi_basename": ref_fits.name,
        "sector": 20,
        "camera": 3,
        "ccd": 3,
        "offset_threshold": 0.02,
        "x_min": 0,
        "y_min": 0,
        "x_max": 64,
        "y_max": 64,
        "shape": [64, 64],
        "groups": [{"group_id": 0, "group_dx": 0.0, "group_dy": 0.0, "n_frames": 1}],
    }
    (event_dir / "cluster_template_job.json").write_text(json.dumps(job), encoding="utf-8")
    pd.DataFrame(
        {
            "path": [str(ref_fits)],
            "group_id": [0],
            "group_dx": [0.0],
            "group_dy": [0.0],
            "wcs_ok": [True],
        }
    ).to_csv(event_dir / "syndiff_ffi_frames.csv", index=False)


class TestDiffHandoffBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.event_dir = self.root / "events" / "s0020_c3_k3"
        ref = self.root / "ref.fits"
        data = np.zeros((64, 64), dtype=np.float32)
        hdu1 = fits.ImageHDU(data=data)
        hdu1.header["NAXIS1"] = 64
        hdu1.header["NAXIS2"] = 64
        hdu1.header["CRPIX1"] = 32.0
        hdu1.header["CRPIX2"] = 32.0
        hdu1.header["CRVAL1"] = 100.0
        hdu1.header["CRVAL2"] = 10.0
        hdu1.header["CDELT1"] = -0.01
        hdu1.header["CDELT2"] = 0.01
        hdu1.header["CTYPE1"] = "RA---TAN"
        hdu1.header["CTYPE2"] = "DEC--TAN"
        fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(ref, overwrite=True)
        self.ref_fits = ref
        _minimal_handoff(self.event_dir, ref)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_handoff_inherits_cluster_crop(self):
        cfg = SynDiffConfig(
            output_dir=str(self.event_dir),
            target_ra=100.0,
            target_dec=10.0,
            pipeline=[{"kind": "shared_mask"}],
        )
        wcs_table, crop, ref, thresh = _load_template_handoff(
            cfg, str(self.event_dir), None
        )
        self.assertEqual(len(wcs_table), 1)
        self.assertEqual(crop["x_max"], 64)
        self.assertEqual(ref, str(self.ref_fits))
        self.assertEqual(thresh, 0.02)

    def test_load_handoff_diff_config_crop_override(self):
        cfg = SynDiffConfig(
            output_dir=str(self.event_dir),
            target_ra=100.0,
            target_dec=10.0,
            crop_mode="target_box",
            crop_box_size=32,
            pipeline=[{"kind": "shared_mask"}],
        )
        _, crop, _, _ = _load_template_handoff(cfg, str(self.event_dir), None)
        self.assertEqual(crop["shape"], (32, 32))

    def test_missing_manifest_raises(self):
        cfg = SynDiffConfig(output_dir=str(self.root / "empty"), pipeline=[])
        with self.assertRaises(RuntimeError):
            _load_template_handoff(cfg, str(self.root / "empty"), None)

    def test_validate_rejects_wcs_grouping_stage(self):
        cfg = SynDiffConfig(
            pipeline=[{"kind": "wcs_grouping"}],
        )
        with self.assertRaises(ValueError) as ctx:
            validate_pipeline(cfg)
        self.assertIn("not a differencing stage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
