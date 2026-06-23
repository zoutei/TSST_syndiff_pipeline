"""Tests for Hotpants calibration metadata and kernel light-curve calibration."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages.hotpants import build_hotpants_config
from syndiff_pipeline.difference_imaging.support.flux_calibration import (
    PHOT_CALIB_COLUMNS,
    TEMPLATE_ZP,
    TESS_JY_ZP,
    apply_kernel_calibration,
    build_phot_calib_rows,
    flux_jy_from_tess_mag,
    kernel_ref_from_kernel_sums,
    kernel_sum_at_center,
    phot_calib_csv_path,
    stamp_diff_calib_metadata,
    tess_mag_from_cts_per_s,
    tess_zp_from_kernel_sum,
    validate_kernel_sum,
    write_phot_calib_table,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    PHOT_CALIB_CSV_BASENAME,
    meta_workspace_dir_from_diffs_dir,
)

S0020_KF_DIR = (
    "/astro/armin/koji/syndiff/workspace/events/s0020_c3_k3_2020ut"
    "/ws_single_hp_kernel/kernel_fit"
)
S0020_REFERENCE_KERNEL_SUM = 0.021490


class TestKernelSumAtCenter(unittest.TestCase):
    def test_delta_kernel_sums_to_one(self):
        kimg = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        with patch(
            "syndiff_pipeline.difference_imaging.stages.kernel.kernel_image_at_coords",
            return_value=kimg,
        ):
            total = kernel_sum_at_center(np.zeros(1), object(), (3, 3))
        self.assertAlmostEqual(total, 1.0)


class TestTessZpFromKernelSum(unittest.TestCase):
    def test_template_anchor_formula(self):
        k = 0.021490
        expected = TEMPLATE_ZP + 2.5 * np.log10(k)
        self.assertAlmostEqual(tess_zp_from_kernel_sum(k), expected)

    def test_s0020_reference_kernel_sum(self):
        tz = tess_zp_from_kernel_sum(S0020_REFERENCE_KERNEL_SUM)
        self.assertLess(tz, TEMPLATE_ZP)
        self.assertAlmostEqual(tz, 20.8306, places=3)


class TestKernelRefFromKernelSums(unittest.TestCase):
    def test_sigma_clip_excludes_outlier(self):
        values = np.array([0.021] * 10 + [0.5])
        ref = kernel_ref_from_kernel_sums(values)
        self.assertAlmostEqual(ref, 0.021, places=3)


class TestDiffCalibMetadataHeader(unittest.TestCase):
    def test_stamp_adds_keywords(self):
        k = S0020_REFERENCE_KERNEL_SUM
        tz = tess_zp_from_kernel_sum(k)
        base = fits.Header()
        base["BUNIT"] = "electrons/s"
        hdr = stamp_diff_calib_metadata(base, k, tz)
        self.assertEqual(hdr["BUNIT"], "electrons/s")
        self.assertAlmostEqual(float(hdr["FLUXSCAL"]), k)
        self.assertAlmostEqual(float(hdr["KERNZPT"]), tz)


class TestHotpantsDoesNotStampDiffHeaders(unittest.TestCase):
    def test_process_one_frame_imports_no_header_stamper(self):
        import syndiff_pipeline.difference_imaging.stages.hotpants as hp_mod

        src = open(hp_mod.__file__, encoding="utf-8").read()
        self.assertNotIn("stamp_diff_calib_metadata", src)
        self.assertNotIn("apply_flux_calibration", src)
        self.assertNotIn("stamp_phot_calib_header", src)
        self.assertNotIn("read_sci_zp_from_ffi", src)


class TestPhotCalibTable(unittest.TestCase):
    def test_write_phot_calib_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            k = 0.021490
            results = [
                {
                    "product_id": "ffi-001",
                    "stem": "hp_d_ffi-001",
                    "kernel_sum": k,
                    "tess_zp": tess_zp_from_kernel_sum(k),
                    "success": True,
                },
                {
                    "product_id": "ffi-002",
                    "success": False,
                },
            ]
            path = write_phot_calib_table(tmp, results)
            self.assertEqual(path, phot_calib_csv_path(tmp))
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(os.path.basename(path), PHOT_CALIB_CSV_BASENAME)
            table = pd.read_csv(path)
            self.assertEqual(list(table.columns), list(PHOT_CALIB_COLUMNS))
            self.assertEqual(len(table), 2)
            self.assertTrue(table.loc[0, "success"])
            self.assertFalse(table.loc[1, "success"])

    def test_build_phot_calib_rows_uses_ffi_product_id(self):
        rows = build_phot_calib_rows(
            [
                {
                    "ffi_product_id": "abc",
                    "kernel_sum": 0.02,
                    "tess_zp": 19.0,
                    "success": True,
                }
            ]
        )
        self.assertEqual(rows[0]["product_id"], "abc")
        self.assertAlmostEqual(rows[0]["tess_zp"], 19.0)


class TestApplyKernelCalibration(unittest.TestCase):
    def test_equalizes_flux_at_same_true_flux(self):
        k1, k2 = 0.02, 0.03
        phot_calib = pd.DataFrame(
            [
                {
                    "product_id": "tess111",
                    "kernel_sum": k1,
                    "tess_zp": tess_zp_from_kernel_sum(k1),
                },
                {
                    "product_id": "tess222",
                    "kernel_sum": k2,
                    "tess_zp": tess_zp_from_kernel_sum(k2),
                },
            ]
        )
        lc_df = pd.DataFrame(
            [
                {
                    "filename": "/ws/hp_d/tess111_hp_d.fits",
                    "flux": 100.0,
                    "eflux": 10.0,
                },
                {
                    "filename": "/ws/hp_d/tess222_hp_d.fits",
                    "flux": 100.0 * k2 / k1,
                    "eflux": 8.0,
                },
            ]
        )
        out = apply_kernel_calibration(lc_df, phot_calib)
        self.assertAlmostEqual(out.loc[0, "kernel_ref"], (k1 + k2) / 2.0)
        self.assertAlmostEqual(out.loc[0, "flux"], out.loc[1, "flux"], places=5)
        self.assertAlmostEqual(out.loc[0, "flux_uncal"], 100.0)
        self.assertAlmostEqual(
            out.loc[0, "eflux"], 10.0 * out.loc[0, "flux"] / 100.0
        )

    def test_tmag_uses_flux_uncal_and_per_epoch_tess_zp(self):
        k = 0.02
        tz = tess_zp_from_kernel_sum(k)
        flux = 100.0
        phot_calib = pd.DataFrame(
            [{"product_id": "tess111", "kernel_sum": k, "tess_zp": tz}]
        )
        lc_df = pd.DataFrame(
            [
                {
                    "filename": "/ws/hp_d/tess111_hp_d.fits",
                    "flux": flux,
                    "eflux": 1.0,
                }
            ]
        )
        out = apply_kernel_calibration(lc_df, phot_calib)
        expected_tmag = tess_mag_from_cts_per_s(flux, tz)
        self.assertAlmostEqual(out.loc[0, "flux_uncal"], flux)
        self.assertAlmostEqual(out.loc[0, "tmag"], expected_tmag)
        expected_jy = flux_jy_from_tess_mag(expected_tmag)
        self.assertAlmostEqual(out.loc[0, "flux_jy"], expected_jy)
        self.assertAlmostEqual(out.loc[0, "flux_jy"], TESS_JY_ZP * 10 ** (-0.4 * expected_tmag))

    def test_uses_kernel_sum_already_in_dataframe(self):
        k1, k2 = 0.02, 0.03
        lc_df = pd.DataFrame(
            [
                {
                    "filename": "/ws/hp_d/tess111_hp_d.fits",
                    "flux": 100.0,
                    "eflux": 10.0,
                    "kernel_sum": k1,
                },
                {
                    "filename": "/ws/hp_d/tess222_hp_d.fits",
                    "flux": 100.0 * k2 / k1,
                    "eflux": 8.0,
                    "kernel_sum": k2,
                },
            ]
        )
        out = apply_kernel_calibration(lc_df, phot_calib=None)
        self.assertAlmostEqual(out.loc[0, "flux"], out.loc[1, "flux"], places=5)
        self.assertTrue(np.isfinite(out.loc[0, "tess_zp"]))

    def test_skips_when_no_calib(self):
        lc_df = pd.DataFrame([{"filename": "a.fits", "flux": 1.0, "eflux": 0.1}])
        out = apply_kernel_calibration(lc_df, phot_calib=None)
        self.assertAlmostEqual(out.loc[0, "flux"], 1.0)
        self.assertAlmostEqual(out.loc[0, "flux_uncal"], 1.0)
        self.assertNotIn("kernel_ref", out.columns)


class TestKernelFitReferenceKernelSum(unittest.TestCase):
    def test_s0020_real_kernel_fit_artifacts(self):
        kf_dir = S0020_KF_DIR
        if not os.path.isdir(kf_dir):
            self.skipTest(f"missing s0020 kernel_fit dir: {kf_dir}")

        meta = json.load(
            open(os.path.join(kf_dir, "kernel_fit_meta.json"), encoding="utf-8")
        )
        data = np.load(os.path.join(kf_dir, "kernel_r2.npz"), allow_pickle=False)
        kernel_solution = np.asarray(data["kernel_solution"], dtype=np.float64).ravel()

        with fits.open(os.path.join(kf_dir, "ffi.fits")) as hdul:
            shape = hdul[0].data.shape

        hp_config = build_hotpants_config(
            replace(HotpantsParams(), hp_bgo=0),
            "/tmp/s0020_kf_test",
            "/tmp/s0020_kf_test",
            "ref",
            write_stamps=False,
        )
        k = kernel_sum_at_center(kernel_solution, hp_config, shape)
        tz = tess_zp_from_kernel_sum(k)

        self.assertEqual(meta["product_id"], "tess2019359002923")
        self.assertAlmostEqual(k, S0020_REFERENCE_KERNEL_SUM, places=4)
        self.assertLess(tz, TEMPLATE_ZP)
        self.assertAlmostEqual(tz, 20.8306, places=2)
        self.assertNotIn("reference_tess_zp", meta)


class TestKernelSubtractNoMeta(unittest.TestCase):
    def test_ks_d_has_no_meta_pair_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            diffs = os.path.join(tmp, "ks_d")
            os.makedirs(diffs)
            meta = meta_workspace_dir_from_diffs_dir(diffs)
            self.assertEqual(os.path.basename(meta), "ks_m")
            self.assertFalse(os.path.isdir(meta))


class TestValidateKernelSum(unittest.TestCase):
    def test_rejects_invalid_kernel_sum(self):
        with self.assertRaises(ValueError):
            validate_kernel_sum(0.0)
        with self.assertRaises(ValueError):
            validate_kernel_sum(float("nan"))


if __name__ == "__main__":
    unittest.main()
