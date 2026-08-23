"""Tests for lazy per-frame sci_bkg loading in hotpants."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages import hotpants


class TestLoadSciBkgCrop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bkg_ws = Path(self._tmp.name) / "ks_b"
        self.bkg_ws.mkdir()
        self.shape = (32, 32)
        self.product_id = "tess2026039233236"

    def tearDown(self):
        self._tmp.cleanup()

    def _write_bkg(self, data: np.ndarray) -> None:
        stem = hotpants.workspace_frame_stem(self.product_id, "ks_b")
        path = self.bkg_ws / f"{stem}.fits"
        fits.writeto(path, data.astype(np.float32), overwrite=True)

    def test_loads_existing_fits(self):
        data = np.ones(self.shape, dtype=np.float64) * 3.5
        self._write_bkg(data)
        out = hotpants._load_sci_bkg_crop(str(self.bkg_ws), self.product_id, self.shape)
        np.testing.assert_array_equal(out, data)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            hotpants._load_sci_bkg_crop(str(self.bkg_ws), self.product_id, self.shape)

    def test_shape_mismatch_raises(self):
        self._write_bkg(np.zeros((16, 16), dtype=np.float64))
        with self.assertRaises(ValueError):
            hotpants._load_sci_bkg_crop(str(self.bkg_ws), self.product_id, self.shape)


    def test_loads_spoc_frame_stem(self):
        ffi_stem = "tess2026039233236-s0020-3-3"
        data = np.ones(self.shape, dtype=np.float64) * 2.0
        stem = hotpants.workspace_frame_stem(ffi_stem, "ks_b")
        path = self.bkg_ws / f"{stem}.fits"
        fits.writeto(path, data.astype(np.float32), overwrite=True)
        out = hotpants._load_sci_bkg_crop(str(self.bkg_ws), ffi_stem, self.shape)
        np.testing.assert_array_equal(out, data)


class TestProcessOneFrameFailsClosedOnMissingBkg(unittest.TestCase):
    """A wired-but-missing sci_bkg must fail the frame, not silently proceed
    with zero background (production incident 2026-08-23: 485/495 S20/C3/K3
    tvwcs hp_d frames looked like successes but had no background removed)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.diffs = root / "ws" / "hp_d"
        self.convolved = root / "ws" / "hp_c"
        self.bkg_ws = root / "ws" / "ks_b"
        self.diffs.mkdir(parents=True)
        self.convolved.mkdir(parents=True)
        self.bkg_ws.mkdir(parents=True)
        self.dirs = hotpants.HotpantsWorkspaceDirs(
            diffs=str(self.diffs), convolved=str(self.convolved)
        )
        self.shape = (16, 16)
        self.product_id = "tess2026039233236"
        self.crop_bounds = {"x_min": 0, "x_max": 16, "y_min": 0, "y_max": 16}

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_bkg_fails_frame_without_calling_hotpants(self):
        hp = HotpantsParams(write_convolved=False, write_bkg=False, write_stamps=False)
        sci = np.ones(self.shape, dtype=np.float64)
        tmpl = np.ones(self.shape, dtype=np.float64) * 0.5
        with (
            patch.object(hotpants, "_load_template_cropped", return_value=tmpl),
            patch.object(hotpants, "_resolve_linear_template_pad", return_value=0),
            patch.object(hotpants, "_load_ffi_cropped", return_value=(sci, np.ones(self.shape))),
            patch.object(hotpants, "run_hotpants_frame") as mock_run,
        ):
            result = hotpants._process_one_frame(
                ffi_path="/fake/ffi.fits",
                product_id=self.product_id,
                group_id=0,
                hp=hp,
                template_path_map={0: "/fake/template.fits"},
                mask=np.zeros(self.shape, dtype=np.uint8),
                crop_bounds=self.crop_bounds,
                ref_stars_xy=np.array([[8.0, 8.0]]),
                dirs=self.dirs,
                round_id=2,
                sci_bkg_ws=str(self.bkg_ws),
            )
        self.assertFalse(result["success"])
        self.assertIn("sci_bkg unavailable", result["error_msg"])
        mock_run.assert_not_called()
        self.assertEqual(list(self.diffs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
