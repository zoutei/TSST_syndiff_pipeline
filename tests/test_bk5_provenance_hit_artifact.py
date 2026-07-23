"""BK-5: provenance_hit must not skip-succeed without a locatable artifact."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages import hotpants as hp_mod
from syndiff_pipeline.difference_imaging.stages import kernel_subtract as ks_mod
from syndiff_pipeline.difference_imaging.stages.hotpants import HotpantsWorkspaceDirs


class TestProvenanceHitRequiresArtifact(unittest.TestCase):
    def test_kernel_subtract_indexed_complete_without_file_falls_through(self):
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
            os.makedirs(diffs_dir)
            ffi_path = os.path.join(tmp, "tess2020057105921-s0001.fits")
            Path(ffi_path).write_bytes(b"")

            ks_mod._kernel_subtract_loky_initializer(
                {
                    "crop_bounds": crop_bounds,
                    "shared_mask": shared_mask,
                    "convolved_table": convolved_table,
                    "phot_box_size": 4,
                    "diffs_dir": diffs_dir,
                    "bkg_dir": None,
                    "diffs_label": "ks_d",
                    "bkg_label": None,
                    "output_dir": tmp,
                    "manifest": pd.DataFrame(),
                    "sck": (1, 1, 1),
                    "data_root": tmp,
                    "workspace_root": tmp,
                    "output_store_name": None,
                    "downsample_fp": "downsample_fp_test",
                    "cfg": None,
                }
            )

            with patch.object(
                ks_mod.provenance_glue,
                "diff_image_complete_in_store",
                return_value=True,
            ), patch(
                "syndiff_pipeline.difference_imaging.orchestration.diff_store"
                ".resolve_diff_write_path",
                return_value=Path(os.path.join(diffs_dir, "missing.fits.fz")),
            ), patch.object(
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
            ), patch.object(
                ks_mod, "_write_image_fits"
            ) as mock_write:
                result = ks_mod._process_one_frame((ffi_path,))

            self.assertTrue(result["success"])
            self.assertFalse(result.get("skipped", False))
            self.assertFalse(result.get("provenance_hit", False))
            mock_write.assert_called()

    def test_hotpants_indexed_complete_without_file_falls_through(self):
        crop_bounds = {
            "x_min": 0,
            "y_min": 0,
            "x_max": 4,
            "y_max": 4,
            "shape": (4, 4),
        }
        with tempfile.TemporaryDirectory() as tmp:
            diffs_dir = os.path.join(tmp, "hp_d")
            conv_dir = os.path.join(tmp, "hp_c")
            os.makedirs(diffs_dir)
            os.makedirs(conv_dir)
            ffi_path = os.path.join(tmp, "tess2020057105921-s0001.fits")
            Path(ffi_path).write_bytes(b"")
            dirs = HotpantsWorkspaceDirs(diffs=diffs_dir, convolved=conv_dir, bkg=None)
            hp = HotpantsParams()
            sci = np.ones((4, 4), dtype=np.float64)
            err = np.ones((4, 4), dtype=np.float64)
            tmpl = np.ones((4, 4), dtype=np.float64)

            with patch.object(
                hp_mod.provenance_glue,
                "diff_image_complete_in_store",
                return_value=True,
            ), patch(
                "syndiff_pipeline.difference_imaging.orchestration.diff_store"
                ".resolve_diff_write_path",
                return_value=Path(os.path.join(diffs_dir, "missing.fits.fz")),
            ), patch.object(
                hp_mod, "_load_ffi_cropped", return_value=(sci, err)
            ), patch.object(
                hp_mod, "_load_template_cropped", return_value=tmpl
            ), patch.object(
                hp_mod.wcs_grouping, "crop_ffi_header", return_value=MagicMock()
            ), patch.object(
                hp_mod,
                "run_hotpants_frame",
                return_value={
                    "success": True,
                    "diff": sci,
                    "noise": err,
                    "mask": np.zeros((4, 4), dtype=bool),
                    "bkg": None,
                    "convolved": None,
                    "kernel_params_arrays": {},
                },
            ), patch.object(
                hp_mod, "write_diff_noise_mask_fits"
            ) as mock_write:
                result = hp_mod._process_one_frame(
                    ffi_path=ffi_path,
                    product_id="2020057105921",
                    group_id=0,
                    hp=hp,
                    template_path_map={0: "/tmp/t.fits"},
                    mask=np.zeros((4, 4), dtype=bool),
                    crop_bounds=crop_bounds,
                    ref_stars_xy=np.array([[1.0, 1.0]]),
                    dirs=dirs,
                    round_id=1,
                    force_rerun=False,
                    sck=(1, 1, 1),
                    data_root=tmp,
                    workspace_root=tmp,
                    downsample_fp="downsample_fp_test",
                )

            self.assertTrue(result["success"])
            self.assertFalse(result.get("skipped", False))
            self.assertFalse(result.get("provenance_hit", False))
            mock_write.assert_called()

    def test_kernel_subtract_background_emit_skips_when_ffi_fp_missing(self):
        """Empty-input mint guard: None helpers → skip emit, do not mint []."""
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
                    "sck": (1, 1, 1),
                    "data_root": tmp,
                    "workspace_root": tmp,
                    "downsample_fp": "ds_fp",
                    "cfg": None,
                }
            )

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
            ), patch.object(
                ks_mod, "_write_image_fits"
            ), patch.object(
                ks_mod.provenance_glue,
                "ffi_input_fingerprint",
                return_value=None,
            ), patch.object(
                ks_mod.provenance_glue,
                "diff_background_input_fingerprints",
                return_value=None,
            ) as mock_bkg_inputs, patch.object(
                ks_mod.provenance_glue,
                "diff_image_input_fingerprints",
                return_value=["ffi", "ds"],
            ), patch.object(
                ks_mod.provenance_glue, "emit_diff_artifact"
            ) as mock_emit:
                result = ks_mod._process_one_frame((ffi_path,))

            self.assertTrue(result["success"])
            mock_bkg_inputs.assert_called()
            kinds = [c.kwargs.get("kind") for c in mock_emit.call_args_list]
            self.assertIn("diff_image", kinds)
            self.assertNotIn("diff_background", kinds)


class TestKernelSubtractDownsampleFromCfg(unittest.TestCase):
    def test_loop_resolves_downsample_via_from_cfg(self):
        cfg = MagicMock()
        cfg.sector = 20
        cfg.camera = 1
        cfg.ccd = 1
        cfg.data_root = "/tmp/data"
        cfg.output_store_name = None
        cfg.workspace_run_id = None
        cfg.oversampling_factor = 1
        cfg.template_store_name = None
        cfg.site_config_dir = None
        cfg.target_ra = None
        cfg.target_dec = None
        cfg.target_name = None

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                ks_mod.provenance_glue,
                "resolve_downsample_fingerprint_from_cfg",
                return_value="from_cfg_fp",
            ) as mock_from_cfg, patch.object(
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
                    diffs_dir=os.path.join(tmp, "ks_d"),
                    diffs_label="ks_d",
                    cfg=cfg,
                )
            mock_from_cfg.assert_called_once_with(cfg)
            self.assertEqual(
                ks_mod._KERNEL_SUBTRACT_LOKY.get("downsample_fp"), "from_cfg_fp"
            )


if __name__ == "__main__":
    unittest.main()
