"""End-to-end synthetic integration tests for star diff stamps and photometry."""

from __future__ import annotations

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
from syndiff_pipeline.star import diff_runner
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.identifiers import ResolvedHost
from syndiff_pipeline.star.mini_downsample import write_star_mini_templates
from syndiff_pipeline.star.windowed_photometry import run_windowed_forced_photometry


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


def _minimal_ctx(tmp: Path, *, crop_size: int = 128) -> StarEventContext:
    crop_bounds = {
        "x_min": 0,
        "y_min": 0,
        "x_max": crop_size,
        "y_max": crop_size,
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
        master_mapping_fits=str(tmp / "mapping" / "master.fits.fz"),
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


def _resolved_host() -> ResolvedHost:
    return ResolvedHost(
        input_kind="gaia",
        input_value=1060421588522505216,
        tic_id=None,
        gaia_source_id=1060421588522505216,
        ra=0.0,
        dec=0.0,
        phot_g_mean_mag=12.0,
        phot_bp_mean_mag=None,
        phot_rp_mean_mag=None,
        resolution_method="test",
        label=None,
    )


class TestStarDiffClosureIntegration(unittest.TestCase):
    def test_multi_frame_subtraction_and_flat_lightcurve(self):
        rng = np.random.default_rng(42)
        crop_size = 128
        host_x, host_y = 64.0, 64.0
        star_flux = 100.0
        background_level = 20.0
        group_dx, group_dy = 0.0, 0.0
        stamp_size = 24
        kernel_margin_px = 64
        product_ids = [
            "tess2026039233236",
            "tess2026039233237",
            "tess2026039233238",
        ]

        hp_config = _identity_hp_config()
        kernel_solution = _identity_kernel_solution(hp_config)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _minimal_ctx(root, crop_size=crop_size)
            (root / "stamps").mkdir()
            (root / "lc").mkdir()

            mini_arrays = np.zeros((1, 3, 11, 11), dtype=np.float32)
            mini_arrays[0, 0, 5, 5] = star_flux
            mini_origin = (int(host_x) - 5, int(host_y) - 5)
            mini_paths = write_star_mini_templates(
                root / "mini",
                mini_arrays,
                offsets=np.array([[group_dx, group_dy]], dtype=np.float64),
                roi_origin=mini_origin,
                host_identifier_metadata={
                    "gaia_source_id": "1060421588522505216",
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

            manifest_rows = []
            noise_by_product: dict[str, np.ndarray] = {}
            for product_id in product_ids:
                noise = rng.normal(0.0, 1.0, size=(crop_size, crop_size))
                noise_by_product[product_id] = noise
                science = (
                    np.full((crop_size, crop_size), background_level, dtype=np.float64)
                    + noise
                )
                conv_temp = (
                    np.full((crop_size, crop_size), background_level, dtype=np.float64)
                    + s_conv_full
                )

                ffi_basename = f"{product_id}-s0020-3-3-0165-s_ffic.fits"
                ffi_path = str(root / ffi_basename)
                _write_raw_ffi(ffi_path, science)

                conv_stem = hotpants.workspace_frame_stem(product_id, "hp_c")
                bkg_stem = hotpants.workspace_frame_stem(product_id, "ks_b_s")
                _write_crop_sized_fits(
                    hotpants.workspace_frame_fits_path(
                        ctx.baseline_convolved_dir, conv_stem
                    ),
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
                manifest_rows.append(
                    {
                        "path": ffi_path,
                        "group_dx": group_dx,
                        "group_dy": group_dy,
                        "group_id": 0,
                    }
                )

            pd.DataFrame(manifest_rows).to_csv(
                Path(ctx.event_dir) / "frames.csv",
                index=False,
            )

            stamp_paths: list[str] = []
            for product_id in product_ids:
                stamp, metadata = diff_runner.compute_star_only_stamp_for_frame(
                    ctx=ctx,
                    product_id=product_id,
                    host_local_xy=(host_x, host_y),
                    mini_template_fits_paths={(group_dx, group_dy): mini_paths[0]},
                    stamp_size=stamp_size,
                    kernel_margin_px=kernel_margin_px,
                )

                wy0 = metadata["window_y0"]
                wx0 = metadata["window_x0"]
                wy1 = wy0 + stamp.shape[0]
                wx1 = wx0 + stamp.shape[1]
                noise_window = noise_by_product[product_id][wy0:wy1, wx0:wx1]

                np.testing.assert_allclose(
                    stamp,
                    noise_window,
                    atol=1e-5,
                    rtol=1e-5,
                )

                stamp_path = str(root / "stamps" / f"{product_id}.fits.fz")
                written = diff_runner.write_star_diff_stamp(
                    stamp_path,
                    stamp,
                    window_origin=(metadata["window_x0"], metadata["window_y0"]),
                    host_local_xy=(host_x, host_y),
                )
                stamp_paths.append(written)

            host = _resolved_host()
            lc_results = run_windowed_forced_photometry(
                stamp_paths,
                host=host,
                methods=[
                    {"name": "ap3", "type": "aperture", "tar_ap": 3, "sky_in": 5, "sky_out": 9}
                ],
                output_dir=str(root / "lc"),
            )
            fluxes = lc_results["ap3"]["flux_wo_sky"].astype(float).values
            self.assertEqual(len(fluxes), len(product_ids))
            self.assertLess(np.nanstd(fluxes), 5.0)
            self.assertLess(np.nanmax(np.abs(fluxes)), 10.0)


if __name__ == "__main__":
    unittest.main()
