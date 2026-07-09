"""Tests for star_segments.py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.identifiers import ResolvedHost
from syndiff_pipeline.star.star_segments import (
    PS1_REMOVED_STARS_CSV,
    find_owning_skycell_for_host,
    isolate_and_write_mini_templates,
    isolate_host_segment,
)
from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
    RELEVANT_WCS_KEYS,
)


def _gaussian_image(
    size: int,
    sources: list[tuple[float, float, float, float]],
    background: float = 1.0,
    uncert: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size]
    data = np.full((size, size), background, dtype=np.float32)
    for cx, cy, amp, sigma in sources:
        data += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    uncert_arr = np.full_like(data, uncert)
    return data, uncert_arr


def _catalog_row(
    *,
    pixel_x: float,
    pixel_y: float,
    tess_mag: float,
    source_id: int,
) -> dict:
    return {
        "source_id": source_id,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "tess_mag": tess_mag,
        "ra": 120.0,
        "dec": 30.0,
        "phot_g_mean_mag": tess_mag,
        "phot_bp_mean_mag": tess_mag + 0.2,
        "phot_rp_mean_mag": tess_mag - 0.2,
    }


def _resolved_host(source_id: int = 1060421588522505216) -> ResolvedHost:
    return ResolvedHost(
        input_kind="gaia",
        input_value=source_id,
        tic_id=None,
        gaia_source_id=source_id,
        ra=120.0,
        dec=30.0,
        phot_g_mean_mag=12.0,
        phot_bp_mean_mag=12.2,
        phot_rp_mean_mag=11.8,
        resolution_method="test",
        label=None,
    )


def _minimal_ctx(tmp: Path, **overrides) -> StarEventContext:
    mapping_rel = Path("skycell_pixel_mapping") / "sector_0020" / "camera_3" / "ccd_2"
    mapping_dir = tmp / "data" / "skycell_pixel_mapping"
    base = StarEventContext(
        target=Target(
            sector=20,
            camera=3,
            ccd=2,
            target_ra=120.0,
            target_dec=30.0,
            target_name="s20_astrometry",
        ),
        event_dir=str(tmp / "event"),
        workspace_root=str(tmp / "workspace"),
        data_root=str(tmp / "data"),
        cluster_job_path=str(tmp / "event" / "cluster_template_job.json"),
        cluster_job={
            "x_min": 0,
            "y_min": 0,
            "x_max": 20,
            "y_max": 20,
            "shape": [20, 20],
            "sector": 20,
            "camera": 3,
            "ccd": 2,
            "groups": [{"group_dx": 0.0, "group_dy": 0.0, "group_id": 0}],
        },
        crop_bounds={"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20, "shape": (20, 20)},
        mapping_dir=str(mapping_dir),
        mapping_csv=str(mapping_dir / "sector_0020" / "camera_3" / "ccd_2" / "tess_s0020_3_2_master_skycells_list.csv"),
        master_mapping_fits=str(
            mapping_dir / "sector_0020" / "camera_3" / "ccd_2" / "tess_s0020_3_2_master_pixels2skycells.fits.gz"
        ),
        gaia_catalog_path=str(tmp / "data" / "catalogs" / "gaia.csv"),
        templates_dir=str(tmp / "templates"),
        reference_ffi_path=str(tmp / "ref.fits"),
        sector=20,
        camera=3,
        ccd=2,
        baseline_workspace_dir=str(tmp / "event" / "ws"),
        baseline_diffs_label="hp_d",
        baseline_convolved_dir=str(tmp / "event" / "ws" / "hp_c"),
        baseline_phot_bkg_dir=str(tmp / "event" / "ws" / "ks_b_s"),
        baseline_phot_bkg_label="ks_b_s",
        baseline_kernels_dir=str(tmp / "event" / "ws" / "hp_d_kernels"),
    )
    for key, val in overrides.items():
        setattr(base, key, val)
    return base


def _make_skycell_wcs_row(name: str, ra: float, dec: float) -> dict:
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [2400.0, 2400.0]
    wcs.wcs.cdelt = [-0.000228, 0.000228]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    row = {"NAME": name, "RA": ra, "DEC": dec}
    for key in RELEVANT_WCS_KEYS:
        row[key] = header[key] if key in header else (1.0 if key.startswith("PC") else 0.0)
    row["NAXIS1"] = 4800
    row["NAXIS2"] = 4800
    return row


class TestIsolateHostSegment(unittest.TestCase):
    def test_isolated_star_segment(self):
        data, uncert = _gaussian_image(
            128,
            [(40.0, 40.0, 300.0, 1.5)],
            background=0.05,
            uncert=0.05,
        )
        host = _resolved_host(source_id=1001)
        catalog = pd.DataFrame(
            [_catalog_row(pixel_x=40.0, pixel_y=40.0, tess_mag=10.0, source_id=1001)]
        )
        mask = np.zeros_like(data, dtype=np.uint16)

        result = isolate_host_segment(
            data,
            uncert,
            mask,
            catalog,
            host,
            (40.0, 40.0),
            sigma=3.0,
        )

        self.assertGreater(result.target_seg_id, 0)
        self.assertFalse(result.blend_flag)
        self.assertTrue(result.blended_catalog_rows.empty)
        self.assertGreater(np.count_nonzero(result.star_only_image), 0)
        self.assertAlmostEqual(
            float(result.star_only_image[40, 40]),
            float(result.background_suppressed[40, 40]),
        )
        self.assertEqual(np.count_nonzero(result.star_only_image), np.count_nonzero(result.background_suppressed))

    def test_blended_segment_flags_neighbor(self):
        data, uncert = _gaussian_image(
            128,
            [
                (50.0, 50.0, 280.0, 2.5),
                (54.0, 52.0, 260.0, 2.5),
            ],
            background=0.05,
            uncert=0.05,
        )
        host = _resolved_host(source_id=2001)
        catalog = pd.DataFrame(
            [
                _catalog_row(pixel_x=50.0, pixel_y=50.0, tess_mag=10.0, source_id=2001),
                _catalog_row(pixel_x=54.0, pixel_y=52.0, tess_mag=11.0, source_id=2002),
            ]
        )
        mask = np.zeros_like(data, dtype=np.uint16)

        result = isolate_host_segment(
            data,
            uncert,
            mask,
            catalog,
            host,
            (50.0, 50.0),
            sigma=2.5,
        )

        self.assertGreater(result.target_seg_id, 0)
        if result.blend_flag:
            self.assertIn(2002, result.blended_catalog_rows["source_id"].tolist())
        self.assertGreater(np.count_nonzero(result.star_only_image), 0)

    def test_no_segment_returns_zeros_without_raising(self):
        data = np.full((32, 32), 0.01, dtype=np.float32)
        uncert = np.full_like(data, 0.01)
        mask = np.zeros_like(data, dtype=np.uint16)
        host = _resolved_host()
        catalog = pd.DataFrame(
            [_catalog_row(pixel_x=16.0, pixel_y=16.0, tess_mag=12.0, source_id=999)]
        )

        result = isolate_host_segment(
            data,
            uncert,
            mask,
            catalog,
            host,
            (16.0, 16.0),
            sigma=5.0,
        )

        self.assertEqual(result.target_seg_id, 0)
        self.assertFalse(result.blend_flag)
        np.testing.assert_array_equal(result.star_only_image, 0.0)


class TestFindOwningSkycellForHost(unittest.TestCase):
    def _write_mapping_fixtures(self, tmp: Path) -> StarEventContext:
        mapping_rel = Path("sector_0020") / "camera_3" / "ccd_2"
        mapping_dir = tmp / "data" / "skycell_pixel_mapping" / mapping_rel
        mapping_dir.mkdir(parents=True)

        tess_map = np.full((20, 20), -1, dtype=np.int32)
        tess_map[7, 7] = 0
        tess_map[0:5, :] = 1
        hdu0 = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(data=tess_map)
        fits.HDUList([hdu0, hdu1]).writeto(
            mapping_dir / "tess_s0020_3_2_master_pixels2skycells.fits.gz",
            overwrite=True,
        )

        rows = [
            _make_skycell_wcs_row("skycell.1.10", 120.0, 30.0),
            _make_skycell_wcs_row("skycell.1.11", 121.0, 31.0),
        ]
        pd.DataFrame(rows).to_csv(
            mapping_dir / "tess_s0020_3_2_master_skycells_list.csv",
            index=False,
        )

        reg_path = mapping_dir / "skycell.1.10_reg.fits.gz"
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros((4, 4), dtype=np.int32))]).writeto(
            reg_path,
            overwrite=True,
        )

        return _minimal_ctx(tmp)

    def test_returns_single_canonical_skycell(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = self._write_mapping_fixtures(tmp)
            host = _resolved_host()

            with patch(
                "syndiff_pipeline.star.star_segments.resolve_host_full_ffi_xy",
                return_value=(7.0, 7.0),
            ), patch(
                "syndiff_pipeline.star.star_segments.world_ra_dec_to_pixel",
                return_value=(2400.0, 2400.0),
            ):
                result = find_owning_skycell_for_host(ctx, host)

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["skycell_name"], "skycell.1.10")

    def test_unmapped_pixel_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = self._write_mapping_fixtures(tmp)
            host = _resolved_host()

            with patch(
                "syndiff_pipeline.star.star_segments.resolve_host_full_ffi_xy",
                return_value=(10.0, 10.0),
            ):
                result = find_owning_skycell_for_host(ctx, host)

            self.assertTrue(result.empty)

    def test_out_of_bounds_pixel_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = self._write_mapping_fixtures(tmp)
            host = _resolved_host()

            with patch(
                "syndiff_pipeline.star.star_segments.resolve_host_full_ffi_xy",
                return_value=(99.0, 99.0),
            ):
                result = find_owning_skycell_for_host(ctx, host)

            self.assertTrue(result.empty)


class TestDebugPlotsGating(unittest.TestCase):
    """``write_debug_plots`` should gate both segmentation and downsample PNGs."""

    def _run_with_mocks(self, tmp: Path, *, write_debug_plots: bool):
        ctx = _minimal_ctx(tmp)
        Path(ctx.event_dir).mkdir(parents=True, exist_ok=True)
        host = _resolved_host()
        output_dir = str(Path(ctx.event_dir) / "star" / str(host.gaia_source_id) / "mini_templates")

        skycell_table = pd.DataFrame(
            [{
                "skycell_name": "skycell.1.10",
                "host_pixel_x": 40.0,
                "host_pixel_y": 40.0,
                "host_in_cell": True,
                "reg_file": str(tmp / "reg.fits.gz"),
            }]
        )

        fake_image = np.full((80, 80), 1.0, dtype=np.float32)
        fake_uncert = np.full((80, 80), 0.1, dtype=np.float32)
        fake_mask = np.zeros((80, 80), dtype=np.uint16)
        fake_wcs = WCS(naxis=2)

        seg_result = SimpleNamespace(
            target_seg_id=3,
            blend_flag=False,
            filled_seg_map=np.zeros((80, 80), dtype=np.int32),
            background_suppressed=fake_image,
            star_only_image=fake_image,
            blended_catalog_rows=pd.DataFrame({"source_id": []}),
        )

        with (
            patch("syndiff_pipeline.star.star_segments._host_already_removed", return_value=False),
            patch("syndiff_pipeline.star.star_segments.resolve_host_full_ffi_xy", return_value=(40.0, 40.0)),
            patch("syndiff_pipeline.star.star_segments.find_owning_skycell_for_host", return_value=skycell_table),
            patch("syndiff_pipeline.star.star_segments._load_gaia_catalog", return_value=pd.DataFrame({"source_id": []})),
            patch(
                "syndiff_pipeline.star.star_segments.load_and_combine_skycell",
                return_value=(fake_image, fake_mask, fake_uncert, fake_wcs),
            ),
            patch("syndiff_pipeline.star.star_segments.project_gaia_to_skycell", return_value=pd.DataFrame()),
            patch("syndiff_pipeline.star.star_segments.isolate_host_segment", return_value=seg_result),
            patch(
                "syndiff_pipeline.star.star_segments.convolve_star_only_cutout",
                return_value=(np.ones((10, 10), dtype=np.float32), (0, 0)),
            ),
            patch(
                "syndiff_pipeline.star.star_segments.offsets_from_cluster_job_payload",
                return_value=np.array([[0.0, 0.0]]),
            ),
            patch("syndiff_pipeline.star.star_segments.load_tess_wcs", return_value=(fake_wcs, None)),
            patch(
                "syndiff_pipeline.star.star_segments._load_skycell_csv",
                return_value=pd.DataFrame({"NAME": ["skycell.1.10"]}),
            ),
            patch("syndiff_pipeline.star.star_segments.precompute_shifts_for_offsets", return_value={}),
            patch("syndiff_pipeline.star.star_segments._reg_file_for_skycell", return_value=str(tmp / "reg.fits.gz")),
            patch("syndiff_pipeline.star.star_segments._full_ffi_mapping_shape", return_value=(80, 80)),
            patch(
                "syndiff_pipeline.star.star_segments.downsample_star_arrays",
                return_value={(0, 0): np.ones((20, 20), dtype=np.float32)},
            ),
            patch(
                "syndiff_pipeline.star.star_segments.write_star_mini_templates",
                return_value=[str(tmp / "mini_0_0.fits.gz")],
            ),
            patch("syndiff_pipeline.star.star_segments.write_ps1_segment_overlay_png") as mock_seg_png,
            patch("syndiff_pipeline.star.star_segments.write_mini_template_downsample_png") as mock_ds_png,
        ):
            mock_seg_png.return_value = str(tmp / "seg.png")
            mock_ds_png.return_value = str(tmp / "ds.png")
            result = isolate_and_write_mini_templates(
                ctx,
                host,
                output_dir=output_dir,
                write_debug_plots=write_debug_plots,
            )
        return result, mock_seg_png, mock_ds_png

    def test_plots_written_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, mock_seg_png, mock_ds_png = self._run_with_mocks(
                Path(tmpdir), write_debug_plots=True
            )
            self.assertEqual(result.get("mini_template_paths"), [str(Path(tmpdir) / "mini_0_0.fits.gz")])
            mock_seg_png.assert_called_once()
            mock_ds_png.assert_called_once()

    def test_plots_skipped_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, mock_seg_png, mock_ds_png = self._run_with_mocks(
                Path(tmpdir), write_debug_plots=False
            )
            self.assertEqual(result.get("mini_template_paths"), [str(Path(tmpdir) / "mini_0_0.fits.gz")])
            mock_seg_png.assert_not_called()
            mock_ds_png.assert_not_called()


class TestAlreadyRemovedShortCircuit(unittest.TestCase):
    def test_skips_when_host_in_removed_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = _minimal_ctx(tmp)
            event_dir = Path(ctx.event_dir)
            event_dir.mkdir(parents=True)
            host = _resolved_host(source_id=1060421588522505216)
            pd.DataFrame({"source_id": [host.gaia_source_id, 999]}).to_csv(
                event_dir / PS1_REMOVED_STARS_CSV,
                index=False,
            )

            result = isolate_and_write_mini_templates(
                ctx,
                host,
                output_dir=str(event_dir / "star" / str(host.gaia_source_id) / "mini_templates"),
            )

            self.assertTrue(result["already_removed"])
            self.assertEqual(result["host_gaia_source_id"], host.gaia_source_id)
            self.assertEqual(result["mini_template_paths"], [])
            self.assertEqual(result["skycells"], {})


if __name__ == "__main__":
    unittest.main()
