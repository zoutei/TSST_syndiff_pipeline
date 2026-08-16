"""Tests for template coverage and cropped template loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from syndiff_pipeline.common.template_coverage import (
    block_sum_oversampled_to_native,
    crop_bounds_subset_of_coverage,
    load_template_count_cropped,
    template_coverage_ffi_bounds,
    template_crop_slices,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import parse_hotpants
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    TemplateCoverageError,
    _load_ffi_cropped,
    _load_template_cropped,
    resolve_hotpants_oversample,
)


class TestTemplateCoverage(unittest.TestCase):
    def _write_template(
        self,
        path: Path,
        shape: tuple[int, int],
        *,
        xmin=0,
        ymin=0,
        oversamp: int = 1,
        native_shape: tuple[int, int] | None = None,
    ) -> None:
        ny, nx = shape
        data = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
        hdu = fits.PrimaryHDU(data=data)
        if native_shape is None:
            native_ny, native_nx = ny // max(1, oversamp), nx // max(1, oversamp)
        else:
            native_ny, native_nx = native_shape
        hdu.header["XMIN"] = xmin
        hdu.header["YMIN"] = ymin
        hdu.header["XMAX"] = xmin + native_nx
        hdu.header["YMAX"] = ymin + native_ny
        hdu.header["MAPGRID"] = 3
        if oversamp > 1:
            hdu.header["OVERSAMP"] = oversamp
        hdu.writeto(path, overwrite=True)

    def test_full_chip_template_smaller_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl.fits"
            self._write_template(p, (2048, 2048))
            cov = template_coverage_ffi_bounds(str(p))
            crop = {
                "x_min": 512,
                "x_max": 1536,
                "y_min": 512,
                "y_max": 1536,
                "shape": (1024, 1024),
            }
            self.assertTrue(crop_bounds_subset_of_coverage(crop, cov))
            arr = _load_template_cropped(str(p), crop)
            self.assertEqual(arr.shape, (1024, 1024))

    def test_oversampled_template_crop_scales_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl_os2.fits"
            self._write_template(
                p,
                (20, 20),
                xmin=5,
                ymin=5,
                oversamp=2,
                native_shape=(10, 10),
            )
            cov = template_coverage_ffi_bounds(str(p))
            self.assertEqual(cov["oversampling_factor"], 2)
            self.assertEqual(cov["shape"], (10, 10))
            crop = {
                "x_min": 7,
                "x_max": 11,
                "y_min": 6,
                "y_max": 9,
                "shape": (3, 4),
            }
            self.assertTrue(crop_bounds_subset_of_coverage(crop, cov))
            y_slice, x_slice = template_crop_slices(str(p), crop)
            self.assertEqual(y_slice, slice(2, 8))
            self.assertEqual(x_slice, slice(4, 12))
            arr = _load_template_cropped(str(p), crop)
            self.assertEqual(arr.shape, (6, 8))

    def test_ffi_cropped_slice_matches_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ffi.fits"
            ny, nx = 512, 512
            sci = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
            err = np.ones((ny, nx), dtype=np.float32)
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(sci, name="SCI"),
                    fits.ImageHDU(err, name="ERR"),
                ]
            ).writeto(p, overwrite=True)
            bounds = {
                "x_min": 100,
                "x_max": 200,
                "y_min": 50,
                "y_max": 150,
                "shape": (100, 100),
            }
            sci_crop, err_crop = _load_ffi_cropped(str(p), bounds)
            self.assertEqual(sci_crop.shape, (100, 100))
            self.assertEqual(err_crop.shape, (100, 100))
            np.testing.assert_array_equal(
                sci_crop, sci[50:150, 100:200].astype(np.float64)
            )

    def test_roi_template_crop_outside_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl.fits"
            self._write_template(p, (512, 512), xmin=100, ymin=100)
            crop = {
                "x_min": 0,
                "x_max": 512,
                "y_min": 0,
                "y_max": 512,
                "shape": (512, 512),
            }
            with self.assertRaises(TemplateCoverageError):
                _load_template_cropped(str(p), crop)

    def test_mapgrid_v2_without_xmin_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl_v2.fits"
            data = np.zeros((8, 8), dtype=np.float32)
            hdu = fits.PrimaryHDU(data=data)
            hdu.header["MAPGRID"] = 2
            hdu.writeto(p, overwrite=True)
            with self.assertRaises(ValueError) as cm:
                template_coverage_ffi_bounds(str(p))
            self.assertIn("MAPGRID=3", str(cm.exception))

    def test_legacy_without_mapgrid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl_legacy.fits"
            data = np.zeros((10, 12), dtype=np.float32)
            fits.PrimaryHDU(data=data).writeto(p, overwrite=True)
            with self.assertRaises(ValueError):
                template_coverage_ffi_bounds(str(p))


class TestBlockSumOversampledCount(unittest.TestCase):
    def test_block_sum_f2(self):
        hr = np.arange(16, dtype=np.int32).reshape(4, 4)
        native = block_sum_oversampled_to_native(hr, 2)
        self.assertEqual(native.shape, (2, 2))
        self.assertEqual(int(native[0, 0]), 0 + 1 + 4 + 5)
        self.assertEqual(int(native[1, 1]), 10 + 11 + 14 + 15)

    def test_load_count_oversampled_returns_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl_os2.fits"
            flux = np.ones((8, 8), dtype=np.float32)
            # Each native pixel's 2x2 HR block has COUNT=1 → native sum=4.
            count = np.ones((8, 8), dtype=np.int32)
            hdu0 = fits.PrimaryHDU()
            hdu0.header["XMIN"] = 0
            hdu0.header["YMIN"] = 0
            hdu0.header["XMAX"] = 4
            hdu0.header["YMAX"] = 4
            hdu0.header["MAPGRID"] = 3
            hdu0.header["OVERSAMP"] = 2
            fits.HDUList(
                [
                    hdu0,
                    fits.ImageHDU(flux, name="FLUX_SUM"),
                    fits.ImageHDU(count, name="COUNT"),
                ]
            ).writeto(p, overwrite=True)
            crop = {
                "x_min": 0,
                "x_max": 4,
                "y_min": 0,
                "y_max": 4,
                "shape": (4, 4),
            }
            got = load_template_count_cropped(str(p), crop)
            self.assertEqual(got.shape, (4, 4))
            np.testing.assert_array_equal(got, 4)


class TestHotpantsOversampleResolve(unittest.TestCase):
    def test_infer_factor_from_shapes(self):
        self.assertEqual(
            resolve_hotpants_oversample((10, 12), (20, 24), None),
            2,
        )

    def test_configured_mismatch_raises(self):
        with self.assertRaises(ValueError):
            resolve_hotpants_oversample((10, 10), (20, 20), 3)

    def test_native_shapes(self):
        self.assertEqual(
            resolve_hotpants_oversample((32, 32), (32, 32), None),
            1,
        )


class TestParseHotpantsStampMode(unittest.TestCase):
    def test_connected_regions_accepted(self):
        hp = parse_hotpants(
            {
                "kind": "hotpants",
                "stamp_mode": "connected_regions",
                "oversample": 2,
                "region_weight": "npix",
            },
            0,
        )
        self.assertEqual(hp.stamp_mode, "connected_regions")
        self.assertEqual(hp.oversample, 2)
        self.assertEqual(hp.region_weight, "npix")

    def test_bad_stamp_mode_raises(self):
        with self.assertRaises(ValueError):
            parse_hotpants({"kind": "hotpants", "stamp_mode": "bogus"}, 0)


class TestConvolveHrSigmaScaling(unittest.TestCase):
    def test_hr_path_scales_sigma_as_one_over_f_squared(self):
        from unittest.mock import patch

        from syndiff_pipeline.difference_imaging.stages.kernel import (
            convolve_template_with_kernel_solution,
        )
        from hotpants import HotpantsConfig

        hp = HotpantsConfig(
            rkernel=1,
            ko=0,
            bgo=0,
            ngauss=1,
            deg_fixe=[0],
            sigma_gauss=[4.0],
            use_pca=False,
        )
        tmpl = np.zeros((4, 4), dtype=np.float64)
        tmpl[1:3, 1:3] = 1.0
        captured = {}

        def _fake_basis(shape, sigma_gauss, deg_fixe):
            captured["sigma_gauss"] = list(sigma_gauss)
            captured["shape"] = shape
            size = shape[0]
            return [np.zeros((size, size), dtype=np.float64)]

        def _fake_apply(tmpl_in, ks, variance, mask, hp_config, basis, oversample=1):
            return np.zeros((tmpl_in.shape[0] // oversample, tmpl_in.shape[1] // oversample)), None, None, None

        with patch(
            "hotpants.pure.kernel.calculate_kernel_basis",
            side_effect=_fake_basis,
        ), patch(
            "hotpants.pure.convolution.apply_kernel",
            side_effect=_fake_apply,
        ):
            out = convolve_template_with_kernel_solution(
                tmpl,
                np.array([0.0, 1.0]),
                hp,
                oversample=2,
                science_shape=(2, 2),
            )
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(captured["sigma_gauss"], [1.0])  # 4.0 / 2^2
        self.assertEqual(captured["shape"], (5, 5))  # 2*(1*2)+1


if __name__ == "__main__":
    unittest.main()
