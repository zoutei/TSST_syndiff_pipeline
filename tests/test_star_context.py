"""Tests for syndiff_pipeline.star.context."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.star.context import (
    StarEventContext,
    StarPrerequisiteError,
    full_ffi_to_crop_local,
    validate_star_prerequisites,
)


def _minimal_ctx(tmp: str, **overrides) -> StarEventContext:
    base = StarEventContext(
        target=Target(
            sector=20,
            camera=3,
            ccd=2,
            target_ra=0.0,
            target_dec=0.0,
            target_name="s20_astrometry",
        ),
        event_dir=str(Path(tmp) / "event"),
        workspace_root=str(Path(tmp) / "workspace"),
        data_root=str(Path(tmp) / "data"),
        cluster_job_path=str(Path(tmp) / "event" / "event_job.json"),
        cluster_job={"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 120, "shape": [100, 100]},
        crop_bounds={"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 120, "shape": (100, 100)},
        mapping_dir=str(Path(tmp) / "data" / "skycell_pixel_mapping"),
        mapping_csv=str(
            Path(tmp) / "data" / "skycell_pixel_mapping" / "sector_0020" / "camera_3" / "ccd_2"
            / "tess_s0020_3_2_master_skycells_list.csv"
        ),
        master_mapping_fits=str(
            Path(tmp) / "data" / "skycell_pixel_mapping" / "sector_0020" / "camera_3" / "ccd_2"
            / "tess_s0020_3_2_master_pixels2skycells.fits.fz"
        ),
        gaia_catalog_path=str(Path(tmp) / "data" / "catalogs" / "gaia.csv"),
        templates_dir=str(Path(tmp) / "templates"),
        reference_ffi_path=str(Path(tmp) / "ref.fits"),
        sector=20,
        camera=3,
        ccd=2,
        baseline_workspace_dir=str(Path(tmp) / "event" / "ws"),
        baseline_diffs_label="hp_d",
        baseline_diffs_dir=str(Path(tmp) / "event" / "ws" / "hp_d"),
        baseline_convolved_dir=str(Path(tmp) / "event" / "ws" / "hp_c"),
        baseline_phot_bkg_dir=str(Path(tmp) / "event" / "ws" / "ks_b_s"),
        baseline_phot_bkg_label="ks_b_s",
        baseline_kernels_dir=str(Path(tmp) / "event" / "ws" / "hp_d_kernels"),
    )
    for key, val in overrides.items():
        setattr(base, key, val)
    return base


class TestStarContext(unittest.TestCase):
    def test_full_ffi_to_crop_local_arithmetic(self):
        ctx = _minimal_ctx("/tmp/unused")
        x_local, y_local = full_ffi_to_crop_local(ctx, 150.5, 75.25)
        self.assertAlmostEqual(x_local, 140.5)
        self.assertAlmostEqual(y_local, 55.25)

    def test_validate_lists_all_missing_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _minimal_ctx(tmp)
            with self.assertRaises(StarPrerequisiteError) as cm:
                validate_star_prerequisites(ctx)
            message = str(cm.exception)
            self.assertIn("event_job.json", message)
            self.assertIn("frames.csv", message)
            self.assertIn("syndiff_template_", message)
            self.assertIn("baseline diff FITS", message)
            self.assertIn("write_convolved: true", message)
            self.assertIn("photutils background FITS", message)
            self.assertIn("write_kernel_solutions: true", message)
            self.assertIn("shared_mask.fits.fz", message)
            self.assertIn("mapping CSV", message)
            self.assertIn("master_pixels2skycells", message)
            self.assertIn("Gaia catalog CSV", message)

    def test_validate_passes_when_all_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp) / "event"
            templates = Path(tmp) / "templates"
            mapping = (
                Path(tmp)
                / "data"
                / "skycell_pixel_mapping"
                / "sector_0020"
                / "camera_3"
                / "ccd_2"
            )
            gaia = Path(tmp) / "data" / "catalogs"
            lane_root = Path(tmp) / "data" / "s0020" / "c3" / "k2" / "diff"
            lane_hp = lane_root / "hp_d"
            lane_hc = lane_root / "hp_c"
            lane_bkg = lane_root / "ks_b_s"
            from syndiff_pipeline.difference_imaging.stages.hotpants import frame_kernels_dir

            kernels_dir = Path(frame_kernels_dir(str(lane_hp)))
            for directory in (
                lane_hp,
                lane_hc,
                lane_bkg,
                kernels_dir,
                templates,
                mapping,
                gaia,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            event.mkdir(parents=True, exist_ok=True)
            (event / "event_job.json").write_text("{}", encoding="utf-8")
            (event / "frames.csv").write_text("product_id\n", encoding="utf-8")
            (templates / "syndiff_template_0.fits").write_bytes(b"")
            (lane_hp / "tess123_hp_d.fits.fz").write_bytes(b"")
            (lane_hc / "tess123_hp_c.fits.fz").write_bytes(b"")
            (lane_bkg / "tess123_ks_b_s.fits.fz").write_bytes(b"")
            (kernels_dir / "tess123_kernel.npz").write_bytes(b"")
            (lane_root / "shared_mask.fits.fz").write_bytes(b"")
            (mapping / "tess_s0020_3_2_master_skycells_list.csv").write_bytes(b"")
            (mapping / "tess_s0020_3_2_master_pixels2skycells.fits.fz").write_bytes(b"")
            (gaia / "gaia.csv").write_bytes(b"")

            ctx = _minimal_ctx(
                tmp,
                baseline_diffs_dir=str(lane_hp),
                baseline_convolved_dir=str(lane_hc),
                baseline_phot_bkg_dir=str(lane_bkg),
                baseline_kernels_dir=str(kernels_dir),
            )
            validate_star_prerequisites(ctx)

    def test_validate_accepts_scc_lane_baseline_diffs(self):
        from syndiff_pipeline.difference_imaging.stages.hotpants import frame_kernels_dir

        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp) / "event"
            templates = Path(tmp) / "templates"
            mapping = (
                Path(tmp)
                / "data"
                / "skycell_pixel_mapping"
                / "sector_0020"
                / "camera_3"
                / "ccd_2"
            )
            gaia = Path(tmp) / "data" / "catalogs"
            lane_root = Path(tmp) / "data" / "s0020" / "c3" / "k2" / "diff"
            lane_hp = lane_root / "hp_d"
            lane_hc = lane_root / "hp_c"
            lane_bkg = lane_root / "ks_b_s"
            kernels_dir = Path(frame_kernels_dir(str(lane_hp)))
            for directory in (lane_hp, lane_hc, lane_bkg, kernels_dir, templates, mapping, gaia):
                directory.mkdir(parents=True, exist_ok=True)

            event.mkdir(parents=True, exist_ok=True)
            (event / "event_job.json").write_text("{}", encoding="utf-8")
            (event / "frames.csv").write_text("product_id\n", encoding="utf-8")
            (templates / "syndiff_template_0.fits").write_bytes(b"")
            (lane_hp / "tess123_hp_d.fits.fz").write_bytes(b"")
            (lane_hc / "tess123_hp_c.fits.fz").write_bytes(b"")
            (lane_bkg / "tess123_ks_b_s.fits.fz").write_bytes(b"")
            (kernels_dir / "tess123_kernel.npz").write_bytes(b"")
            (lane_root / "shared_mask.fits.fz").write_bytes(b"")
            (mapping / "tess_s0020_3_2_master_skycells_list.csv").write_bytes(b"")
            (mapping / "tess_s0020_3_2_master_pixels2skycells.fits.fz").write_bytes(b"")
            (gaia / "gaia.csv").write_bytes(b"")

            ctx = _minimal_ctx(
                tmp,
                baseline_diffs_dir=str(lane_hp),
                baseline_convolved_dir=str(lane_hc),
                baseline_phot_bkg_dir=str(lane_bkg),
                baseline_kernels_dir=str(kernels_dir),
            )
            validate_star_prerequisites(ctx)

    def test_resolve_baseline_label_dir_prefers_scc_lane(self):
        from syndiff_pipeline.star.context import _resolve_baseline_label_dir

        with tempfile.TemporaryDirectory() as tmp:
            target = Target(
                sector=20,
                camera=3,
                ccd=2,
                target_ra=0.0,
                target_dec=0.0,
                target_name="t",
            )
            event = Path(tmp) / "event"
            ws_hp = event / "ws" / "hp_d"
            ws_hp.mkdir(parents=True)
            (ws_hp / "legacy.fits.fz").write_bytes(b"")
            lane = Path(tmp) / "data" / "s0020" / "c3" / "k2" / "diff" / "hp_d"
            lane.mkdir(parents=True)
            (lane / "lane.fits.fz").write_bytes(b"")
            resolved = _resolve_baseline_label_dir(
                data_root=str(Path(tmp) / "data"),
                target=target,
                event_dir=str(event),
                label="hp_d",
                baseline_run_id="none",
                output_store_name=None,
            )
            self.assertEqual(Path(resolved).resolve(), lane.resolve())


if __name__ == "__main__":
    unittest.main()
