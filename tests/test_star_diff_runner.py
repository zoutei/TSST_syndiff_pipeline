"""Tests for syndiff_pipeline.star.diff_runner."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from hotpants import HotpantsConfig

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.stages import hotpants
from syndiff_pipeline.difference_imaging.stages.kernel import (
    convolve_template_with_kernel_solution,
)
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star import diff_runner
from syndiff_pipeline.star.mini_downsample import write_star_mini_templates


def _identity_hp_config() -> HotpantsConfig:
    return HotpantsConfig(
        rkernel=2,
        ko=0,
        bgo=0,
        ngauss=1,
        deg_fixe=[0],
        sigma_gauss=[1.0],
        use_pca=False,
    )


def _identity_kernel_solution(hp_config: HotpantsConfig) -> np.ndarray:
    ks = np.zeros(hp_config.n_comp_total + 1, dtype=np.float64)
    ks[1] = 1.0
    return ks


def _write_crop_sized_fits(path: str, data: np.ndarray) -> None:
    hotpants._write_image_fits(path, data)


def _write_raw_ffi(path: str, data: np.ndarray) -> None:
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=np.asarray(data, dtype=np.float32))
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


def _minimal_ctx(
    tmp: Path,
    *,
    crop_size: int = 128,
    crop_offset: tuple[int, int] = (0, 0),
) -> StarEventContext:
    ox, oy = crop_offset
    crop_bounds = {
        "x_min": ox,
        "y_min": oy,
        "x_max": ox + crop_size,
        "y_max": oy + crop_size,
        "shape": (crop_size, crop_size),
    }
    event = tmp / "event"
    ws = event / "ws"
    for sub in ("hp_c", "ks_b_s", "hp_d_kernels"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    return StarEventContext(
        target=Target(
            sector=20,
            camera=3,
            ccd=2,
            target_ra=0.0,
            target_dec=0.0,
            target_name="s20_astrometry",
        ),
        event_dir=str(event),
        workspace_root=str(tmp / "workspace"),
        data_root=str(tmp / "data"),
        cluster_job_path=str(event / "cluster_template_job.json"),
        cluster_job=crop_bounds,
        crop_bounds=crop_bounds,
        mapping_dir=str(tmp / "mapping"),
        mapping_csv=str(tmp / "mapping" / "map.csv"),
        master_mapping_fits=str(tmp / "mapping" / "master.fits.gz"),
        gaia_catalog_path=str(tmp / "gaia.csv"),
        templates_dir=str(tmp / "templates"),
        reference_ffi_path=str(tmp / "ref.fits"),
        sector=20,
        camera=3,
        ccd=2,
        baseline_workspace_dir=str(ws),
        baseline_diffs_label="hp_d",
        baseline_convolved_dir=str(ws / "hp_c"),
        baseline_phot_bkg_dir=str(ws / "ks_b_s"),
        baseline_phot_bkg_label="ks_b_s",
        baseline_kernels_dir=str(ws / "hp_d_kernels"),
    )


class TestPlaceMiniTemplateInWindow(unittest.TestCase):
    def test_full_overlap(self):
        mini = np.ones((4, 4), dtype=np.float64)
        out = diff_runner.place_mini_template_in_window(
            mini,
            mini_xmin=10,
            mini_ymin=20,
            window_x0=10,
            window_y0=20,
            window_shape=(4, 4),
        )
        np.testing.assert_allclose(out, mini)

    def test_partial_overlap(self):
        mini = np.arange(16, dtype=np.float64).reshape(4, 4)
        out = diff_runner.place_mini_template_in_window(
            mini,
            mini_xmin=8,
            mini_ymin=8,
            window_x0=10,
            window_y0=10,
            window_shape=(4, 4),
        )
        expected = np.zeros((4, 4), dtype=np.float64)
        expected[0, 0] = mini[2, 2]
        expected[0, 1] = mini[2, 3]
        expected[1, 0] = mini[3, 2]
        expected[1, 1] = mini[3, 3]
        np.testing.assert_allclose(out, expected)

    def test_zero_overlap(self):
        mini = np.ones((3, 3), dtype=np.float64)
        out = diff_runner.place_mini_template_in_window(
            mini,
            mini_xmin=0,
            mini_ymin=0,
            window_x0=20,
            window_y0=20,
            window_shape=(4, 4),
        )
        np.testing.assert_allclose(out, 0.0)

    def test_oversample_scales_canvas_and_origin(self):
        # Native mini origin (2, 3), F=2 → HR origin (4, 6); native 2x2 window → 4x4 canvas.
        mini = np.arange(16, dtype=np.float64).reshape(4, 4)
        out = diff_runner.place_mini_template_in_window(
            mini,
            mini_xmin=2,
            mini_ymin=3,
            window_x0=2,
            window_y0=3,
            window_shape=(2, 2),
            oversample=2,
        )
        self.assertEqual(out.shape, (4, 4))
        np.testing.assert_allclose(out, mini)


class TestWriteStarMiniTemplateOversamp(unittest.TestCase):
    def test_writes_oversamp_and_native_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = np.ones((1, 3, 8, 8), dtype=np.float32)
            paths = write_star_mini_templates(
                tmp,
                arrays,
                offsets=np.array([[0.0, 0.0]]),
                roi_origin=(10, 20),
                host_identifier_metadata={
                    "gaia_source_id": "1",
                    "sector": 20,
                    "camera": 3,
                    "ccd": 2,
                },
                oversampling_factor=2,
            )
            flux, xmin, ymin, os_factor = diff_runner.load_mini_template_flux_sum(paths[0])
            self.assertEqual(os_factor, 2)
            self.assertEqual(xmin, 10)
            self.assertEqual(ymin, 20)
            self.assertEqual(flux.shape, (8, 8))
            with fits.open(paths[0]) as hdul:
                self.assertEqual(int(hdul[0].header["XMAX"]), 14)
                self.assertEqual(int(hdul[0].header["YMAX"]), 24)
                self.assertEqual(int(hdul[0].header["OVERSAMP"]), 2)


class TestWriteStarDiffStamp(unittest.TestCase):
    def test_header_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stamp.fits.gz")
            stamp = np.ones((8, 8), dtype=np.float32)
            diff_runner.write_star_diff_stamp(
                path,
                stamp,
                window_origin=(40, 50),
                host_local_xy=(52.5, 61.25),
            )
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                self.assertEqual(int(hdr["XMIN"]), 40)
                self.assertEqual(int(hdr["YMIN"]), 50)
                self.assertAlmostEqual(float(hdr["HOSTX"]), 12.5)
                self.assertAlmostEqual(float(hdr["HOSTY"]), 11.25)
                np.testing.assert_allclose(hdul[0].data, stamp)


class TestComputeStarOnlyStamp(unittest.TestCase):
    def test_real_convolution_cancels_star(self):
        hp_config = _identity_hp_config()
        kernel_solution = _identity_kernel_solution(hp_config)

        host_y, host_x = 32, 32
        star_flux = 80.0
        background_level = 10.0
        noise = np.random.default_rng(0).normal(0.0, 0.5, size=(64, 64))

        full = np.full((64, 64), background_level, dtype=np.float64) + noise
        full[host_y, host_x] += star_flux

        mini = np.zeros((9, 9), dtype=np.float64)
        mini[4, 4] = star_flux
        mini_xmin = host_x - 4
        mini_ymin = host_y - 4

        stamp_size = 16
        half = stamp_size // 2
        stamp_x0 = host_x - half
        stamp_y0 = host_y - half
        margin = 8
        conv_x0 = stamp_x0 - margin
        conv_y0 = stamp_y0 - margin
        conv_shape = (stamp_size + 2 * margin, stamp_size + 2 * margin)

        mini_embedded = diff_runner.place_mini_template_in_window(
            mini,
            mini_xmin=mini_xmin,
            mini_ymin=mini_ymin,
            window_x0=conv_x0,
            window_y0=conv_y0,
            window_shape=conv_shape,
        )

        s_conv_full = convolve_template_with_kernel_solution(
            mini_embedded, kernel_solution, hp_config
        )
        crop_y0 = stamp_y0 - conv_y0
        crop_x0 = stamp_x0 - conv_x0
        s_conv_stamp = s_conv_full[
            crop_y0 : crop_y0 + stamp_size, crop_x0 : crop_x0 + stamp_size
        ]

        noise_window = noise[
            stamp_y0 : stamp_y0 + stamp_size, stamp_x0 : stamp_x0 + stamp_size
        ]
        conv_temp_window = background_level + s_conv_stamp + noise_window
        ffi_window = background_level + noise_window
        background_window = np.zeros((stamp_size, stamp_size), dtype=np.float64)

        stamp = diff_runner.compute_star_only_stamp(
            ffi_window=ffi_window,
            conv_temp_window=conv_temp_window,
            background_window=background_window,
            mini_star_template_window=mini_embedded,
            kernel_solution=kernel_solution,
            hp_config=hp_config,
            convolve_shape=conv_shape,
            stamp_offset_in_conv=(crop_y0, crop_x0),
        )

        host_in_stamp_y = host_y - stamp_y0
        host_in_stamp_x = host_x - stamp_x0
        self.assertAlmostEqual(
            stamp[host_in_stamp_y, host_in_stamp_x],
            noise_window[host_in_stamp_y, host_in_stamp_x],
            delta=3.0,
        )


class TestComputeStarOnlyStampForFrame(unittest.TestCase):
    def test_end_to_end_subtracts_injected_star(self):
        rng = np.random.default_rng(1)
        crop_size = 128
        host_x, host_y = 64.0, 64.0
        star_flux = 100.0
        background_level = 20.0
        product_id = "tess2026039233236"
        group_dx, group_dy = 0.0, 0.0

        hp_config = _identity_hp_config()
        kernel_solution = _identity_kernel_solution(hp_config)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _minimal_ctx(root, crop_size=crop_size)

            mini_arrays = np.zeros((1, 3, 11, 11), dtype=np.float32)
            mini_arrays[0, 0, 5, 5] = star_flux
            mini_origin = (int(host_x) - 5, int(host_y) - 5)
            mini_paths = write_star_mini_templates(
                root / "mini",
                mini_arrays,
                offsets=np.array([[group_dx, group_dy]], dtype=np.float64),
                roi_origin=mini_origin,
                host_identifier_metadata={
                    "gaia_source_id": "123",
                    "sector": 20,
                    "camera": 3,
                    "ccd": 2,
                },
            )

            mini_flux, mini_xmin, mini_ymin, _os = diff_runner.load_mini_template_flux_sum(
                mini_paths[0]
            )
            mini_full = diff_runner.place_mini_template_in_window(
                mini_flux,
                mini_xmin=mini_xmin,
                mini_ymin=mini_ymin,
                window_x0=0,
                window_y0=0,
                window_shape=(crop_size, crop_size),
            )
            s_conv_full = convolve_template_with_kernel_solution(
                mini_full, kernel_solution, hp_config
            )
            conv_temp = np.full((crop_size, crop_size), background_level, dtype=np.float64)
            conv_temp += s_conv_full

            noise = rng.normal(0.0, 1.0, size=(crop_size, crop_size))
            science = np.full((crop_size, crop_size), background_level, dtype=np.float64) + noise

            ffi_basename = f"{product_id}-s0020-3-3-0165-s_ffic.fits"
            ffi_path = str(root / ffi_basename)
            _write_raw_ffi(ffi_path, science)

            manifest = pd.DataFrame(
                [
                    {
                        "path": ffi_path,
                        "group_dx": group_dx,
                        "group_dy": group_dy,
                        "group_id": 0,
                    }
                ]
            )
            manifest.to_csv(
                Path(ctx.event_dir) / "frames.csv",
                index=False,
            )

            conv_label = "hp_c"
            bkg_label = "ks_b_s"
            conv_stem = hotpants.workspace_frame_stem(product_id, conv_label)
            bkg_stem = hotpants.workspace_frame_stem(product_id, bkg_label)
            _write_crop_sized_fits(
                hotpants.workspace_frame_fits_path(ctx.baseline_convolved_dir, conv_stem),
                conv_temp.astype(np.float32),
            )
            _write_crop_sized_fits(
                hotpants.workspace_frame_fits_path(ctx.baseline_phot_bkg_dir, bkg_stem),
                np.zeros((crop_size, crop_size), dtype=np.float32),
            )
            hotpants.write_frame_kernel_npz(
                ctx.baseline_kernels_dir,
                product_id,
                kernel_solution,
                hp_config,
            )

            stamp, metadata = diff_runner.compute_star_only_stamp_for_frame(
                ctx=ctx,
                product_id=product_id,
                host_local_xy=(host_x, host_y),
                mini_template_fits_paths={(group_dx, group_dy): mini_paths[0]},
                stamp_size=24,
                kernel_margin_px=64,
            )

            self.assertEqual(metadata["product_id"], product_id)
            self.assertAlmostEqual(metadata["group_dx"], group_dx)
            self.assertAlmostEqual(metadata["group_dy"], group_dy)

            host_stamp_y = int(round(host_y)) - metadata["window_y0"]
            host_stamp_x = int(round(host_x)) - metadata["window_x0"]
            residual = stamp[host_stamp_y, host_stamp_x]
            self.assertAlmostEqual(
                residual,
                noise[int(host_y), int(host_x)],
                delta=3.0,
            )
            np.testing.assert_allclose(stamp, noise[
                metadata["window_y0"]:metadata["window_y0"] + stamp.shape[0],
                metadata["window_x0"]:metadata["window_x0"] + stamp.shape[1],
            ], atol=3.0)


class TestLoadFrameKernelForDiff(unittest.TestCase):
    def test_wraps_load_frame_kernel(self):
        hp_config = _identity_hp_config()
        kernel_solution = _identity_kernel_solution(hp_config)
        with tempfile.TemporaryDirectory() as tmp:
            kernels_dir = Path(tmp) / "kernels"
            product_id = "tess2026039233236"
            hotpants.write_frame_kernel_npz(
                str(kernels_dir),
                product_id,
                kernel_solution,
                hp_config,
            )
            loaded_ks, loaded_cfg = diff_runner.load_frame_kernel_for_diff(
                str(kernels_dir), product_id
            )
            np.testing.assert_allclose(loaded_ks, kernel_solution)
            self.assertEqual(int(loaded_cfg.rkernel), int(hp_config.rkernel))


if __name__ == "__main__":
    unittest.main()
