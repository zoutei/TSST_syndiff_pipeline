"""Tests for kernel_subtract (uncalibrated algebraic diff + photutils bkg)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages import kernel_subtract as ks_mod
from syndiff_pipeline.difference_imaging.support.paths import (
    meta_workspace_dir_from_diffs_dir,
)


class TestKernelSubtractUncalibrated(unittest.TestCase):
    def test_writes_raw_diff_and_bkg_without_meta_dir(self):
        crop_bounds = {
            "x_min": 0,
            "y_min": 0,
            "x_max": 4,
            "y_max": 4,
            "shape": (4, 4),
        }
        ffi = np.full((4, 4), 100.0, dtype=np.float64)
        convolved = np.full((4, 4), 40.0, dtype=np.float64)
        phot_bkg = np.full((4, 4), 0.5, dtype=np.float64)
        shared_mask = np.zeros((4, 4), dtype=bool)
        convolved_table = pd.DataFrame(
            [{"group_dx": 0.0, "group_dy": 0.0, "convolved_path": "/tmp/c.fits"}]
        )

        with tempfile.TemporaryDirectory() as tmp:
            diffs_dir = os.path.join(tmp, "ks_d")
            bkg_dir = os.path.join(tmp, "ks_b")
            ffi_path = os.path.join(tmp, "tess2020057105921-s0001.fits")

            with patch.object(
                ks_mod, "resolve_template_for_ffi", return_value=(0.0, 0.0, "/t.fits")
            ), patch.object(
                ks_mod, "lookup_convolved_path", return_value="/tmp/c.fits"
            ), patch.object(
                ks_mod, "_load_ffi_cropped", return_value=(ffi, None)
            ), patch.object(
                ks_mod, "_load_convolved_crop", return_value=convolved
            ), patch.object(
                ks_mod.wcs_grouping, "crop_ffi_header", return_value=None
            ), patch.object(
                ks_mod, "photutils_background_masked", return_value=phot_bkg
            ) as mock_phot, patch.object(
                ks_mod, "_write_image_fits"
            ) as mock_write:
                ks_mod._kernel_subtract_loky_initializer(
                    {
                        "crop_bounds": crop_bounds,
                        "shared_mask": shared_mask,
                        "convolved_table": convolved_table,
                        "phot_box_size": 4,
                        "diffs_dir": diffs_dir,
                        "bkg_dir": bkg_dir,
                        "diffs_label": "ks_d",
                        "bkg_label": "ks_b",
                        "output_dir": tmp,
                        "manifest": pd.DataFrame(),
                    }
                )
                result = ks_mod._process_one_frame((ffi_path,))

            self.assertTrue(result["success"])
            mock_phot.assert_called_once()
            np.testing.assert_array_equal(mock_phot.call_args[0][0], ffi - convolved)
            self.assertEqual(mock_write.call_count, 2)
            diff_data = mock_write.call_args_list[0][0][1]
            np.testing.assert_array_equal(diff_data, ffi - convolved)
            meta = meta_workspace_dir_from_diffs_dir(diffs_dir)
            self.assertFalse(os.path.exists(meta))

    def test_loop_does_not_write_phot_calib(self):
        with tempfile.TemporaryDirectory() as tmp:
            diffs_dir = os.path.join(tmp, "ks_d")
            with patch.object(
                ks_mod, "_process_one_frame", return_value={"success": True, "product_id": "x"}
            ):
                ks_mod.kernel_subtract_loop(
                    ffi_paths=["/fake.fits"],
                    output_dir=tmp,
                    manifest=pd.DataFrame(),
                    crop_bounds={"shape": (2, 2)},
                    shared_mask=np.zeros((2, 2), dtype=bool),
                    convolved_table=pd.DataFrame(),
                    phot_box_size=4,
                    diffs_dir=diffs_dir,
                    diffs_label="ks_d",
                )
            meta = meta_workspace_dir_from_diffs_dir(diffs_dir)
            self.assertFalse(os.path.exists(os.path.join(meta, "phot_calib.csv")))


if __name__ == "__main__":
    unittest.main()
