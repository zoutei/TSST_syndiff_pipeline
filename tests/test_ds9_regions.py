"""Tests for DS9 target region files."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.support.ds9_regions import (
    crop_local_to_ds9_xy,
    format_targets_ds9_regions,
    iter_target_ds9_circles,
    primary_target_region_label,
    write_targets_ds9_regions,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    TARGETS_DS9_REGION_BASENAME,
)


class TestCropLocalToDs9(unittest.TestCase):
    def test_adds_one_for_ds9(self):
        x, y = crop_local_to_ds9_xy(10.5, 20.25)
        self.assertAlmostEqual(x, 11.5)
        self.assertAlmostEqual(y, 21.25)


class TestPrimaryTargetRegionLabel(unittest.TestCase):
    def test_name_and_scc(self):
        self.assertEqual(
            primary_target_region_label("2020ut", 20, 3, 3),
            "2020ut_s0020_c3_k3",
        )


class TestFormatTargetsDs9Regions(unittest.TestCase):
    def test_primary_green_extra_blue_and_roi_comment(self):
        bounds = {"x_min": 100, "y_min": 200, "shape": (1024, 1024)}
        text = format_targets_ds9_regions(
            [
                ("2020ut_s0020_c3_k3", 512.0, 513.0, "green"),
                ("offset_top", 512.0, 506.0, "blue"),
            ],
            crop_bounds=bounds,
            circle_radius_px=10.0,
        )
        self.assertIn("crop-local image coords", text)
        self.assertIn("x_min=100", text)
        self.assertIn("image", text)
        self.assertIn("color=green", text)
        self.assertIn("color=blue", text)
        self.assertIn("text={2020ut_s0020_c3_k3}", text)
        self.assertIn("circle(512.0000,513.0000,10.00)", text)


class TestIterTargetDs9Circles(unittest.TestCase):
    def test_offset_targets_use_crop_local_plus_one(self):
        wcs = pd.DataFrame({"path": ["a.fits", "b.fits"]})
        crop_bounds = {"x_min": 50.0, "y_min": 60.0, "shape": (100, 100)}
        primary_xy = np.array([[10.0, 20.0], [12.0, 22.0]], dtype=np.float64)

        class _Ph:
            @staticmethod
            def per_frame_target_crop_xy(*_a, **_k):
                return primary_xy

            @staticmethod
            def resolve_forced_target_xy(spec, primary, *_a, **_k):
                off = np.array([float(spec["dx"]), float(spec["dy"])])
                return primary + off

        import syndiff_pipeline.difference_imaging.support.ds9_regions as mod

        orig_ph = mod.photometry
        mod.photometry = _Ph()
        try:
            circles = iter_target_ds9_circles(
                target_ra=1.0,
                target_dec=2.0,
                additional_forced_targets=[
                    {"name": "offset_top", "position_mode": "offset", "dx": 0, "dy": -7}
                ],
                wcs_table=wcs,
                crop_bounds=crop_bounds,
                ref_ffi_path="a.fits",
                primary_label="2020ut_s0020_c3_k3",
            )
        finally:
            mod.photometry = orig_ph

        self.assertEqual(len(circles), 2)
        self.assertEqual(circles[0][0], "2020ut_s0020_c3_k3")
        self.assertEqual(circles[0][3], "green")
        self.assertAlmostEqual(circles[0][1], 11.0)
        self.assertAlmostEqual(circles[0][2], 21.0)
        self.assertEqual(circles[1][0], "offset_top")
        self.assertEqual(circles[1][3], "blue")
        # ref row 0: primary (10,20) + (0,-7) → crop (10,13) → ds9 (11, 14)
        self.assertAlmostEqual(circles[1][1], 11.0)
        self.assertAlmostEqual(circles[1][2], 14.0)


class TestWriteTargetsDs9Regions(unittest.TestCase):
    def test_writes_event_root_targets_reg(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp)
            circles = [("2020ut_s0020_c3_k3", 100.0, 200.0, "green")]

            import syndiff_pipeline.difference_imaging.support.ds9_regions as mod

            orig_iter = mod.iter_target_ds9_circles
            mod.iter_target_ds9_circles = lambda **_k: circles
            try:
                out = write_targets_ds9_regions(
                    str(event_dir),
                    target_ra=1.0,
                    target_dec=2.0,
                    target_name="2020ut",
                    sector=20,
                    camera=3,
                    ccd=3,
                    additional_forced_targets=[],
                    wcs_table=pd.DataFrame(),
                    crop_bounds={"x_min": 0, "y_min": 0, "shape": (10, 10)},
                    ref_ffi_path="x.fits",
                )
            finally:
                mod.iter_target_ds9_circles = orig_iter

            self.assertEqual(out, str(event_dir / TARGETS_DS9_REGION_BASENAME))
            self.assertTrue((event_dir / TARGETS_DS9_REGION_BASENAME).is_file())
            body = (event_dir / TARGETS_DS9_REGION_BASENAME).read_text()
            self.assertIn("color=green", body)
            self.assertIn("2020ut_s0020_c3_k3", body)


if __name__ == "__main__":
    unittest.main()
