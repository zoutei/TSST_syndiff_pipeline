"""Tests for PS1 coverage masking via template COUNT extension."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.template_coverage import load_template_count_cropped
from syndiff_pipeline.difference_imaging.stages import masking
from syndiff_pipeline.difference_imaging.stages.masking import ps1_coverage_mask
from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
    tess_straps_csv,
)

# Sector 20 / camera 2 / CCD 1: southern half has COUNT==0 outside PS1 survey.
REAL_TEMPLATE_PATH = Path(
    "/astro/armin/koji/syndiff/data/shifted_downsampled/"
    "sector0020_camera2_ccd1/"
    "syndiff_template_s0020_2_1_dx0.000_dy0.000.fits.gz"
)


class TestPs1CoverageMaskSynthetic(unittest.TestCase):
    def _write_syndiff_template(self, path: Path, count: np.ndarray) -> None:
        hdr = fits.Header()
        hdr["XMIN"] = 0
        hdr["YMIN"] = 0
        hdr["XMAX"] = count.shape[1]
        hdr["YMAX"] = count.shape[0]
        flux = np.ones_like(count, dtype=np.float32)
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(header=hdr),
                fits.ImageHDU(flux, header=hdr, name="FLUX_SUM"),
                fits.ImageHDU(count.astype(np.int32), header=hdr, name="COUNT"),
            ]
        )
        hdul.writeto(path, overwrite=True)

    def test_load_template_count_cropped(self):
        count = np.array([[6000, 4000], [0, 7000]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl.fits"
            self._write_syndiff_template(p, count)
            crop = {
                "x_min": 0,
                "x_max": 2,
                "y_min": 0,
                "y_max": 2,
                "shape": (2, 2),
            }
            got = load_template_count_cropped(str(p), crop)
            np.testing.assert_array_equal(got, count)

    def test_ps1_coverage_mask_threshold(self):
        count = np.array([[6000, 4999, 0]], dtype=np.int32)
        flagged = ps1_coverage_mask(count, min_hit_count=5000)
        np.testing.assert_array_equal(flagged, [[False, True, True]])

    def test_make_shared_mask_sets_bit_16(self):
        count = np.full((8, 8), 6000, dtype=np.int32)
        count[0:3, :] = 0
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tmpl.fits"
            self._write_syndiff_template(p, count)
            crop = {
                "x_min": 0,
                "x_max": 8,
                "y_min": 0,
                "y_max": 8,
                "shape": (8, 8),
            }
            ref = np.zeros((8, 8), dtype=np.float64)
            gaia = pd.DataFrame({"x": [], "y": [], "mag": []})
            mask = masking.make_shared_mask(
                ref_image=ref,
                gaia_df=gaia,
                crop_bounds=crop,
                straps_csv=str(tess_straps_csv()),
                maglim=99.0,
                strapsize=0,
                template_path=str(p),
                ps1_min_hit_count=5000,
            )
            self.assertTrue(np.all((mask[0:3, :] & 16) > 0))
            self.assertTrue(np.all((mask[3:, :] & 16) == 0))


@unittest.skipUnless(
    REAL_TEMPLATE_PATH.is_file(),
    f"real template not available at {REAL_TEMPLATE_PATH}",
)
class TestPs1CoverageMaskRealTemplate(unittest.TestCase):
    def test_real_template_count_crop_matches_survey_edge(self):
        # Lower 1024 rows are outside PS1 survey for this chip.
        crop = {
            "x_min": 556,
            "x_max": 1580,
            "y_min": 1054,
            "y_max": 2078,
            "shape": (1024, 1024),
        }
        count = load_template_count_cropped(str(REAL_TEMPLATE_PATH), crop)
        self.assertIsNotNone(count)
        assert count is not None
        self.assertEqual(count.shape, (1024, 1024))

        n_zero = int((count == 0).sum())
        n_low = int((count < 5000).sum())
        self.assertGreater(n_zero, 0, "expected zero-count pixels in real template crop")
        self.assertEqual(n_zero, n_low)

        flagged = ps1_coverage_mask(count, min_hit_count=5000)
        self.assertEqual(int(flagged.sum()), n_low)

    def test_make_shared_mask_on_real_template(self):
        crop = {
            "x_min": 556,
            "x_max": 1580,
            "y_min": 1054,
            "y_max": 2078,
            "shape": (1024, 1024),
        }
        count = load_template_count_cropped(str(REAL_TEMPLATE_PATH), crop)
        assert count is not None
        expected = ps1_coverage_mask(count, min_hit_count=5000)

        ref = np.zeros((1024, 1024), dtype=np.float64)
        gaia = pd.DataFrame({"x": [], "y": [], "mag": []})
        mask = masking.make_shared_mask(
            ref_image=ref,
            gaia_df=gaia,
            crop_bounds=crop,
            straps_csv=str(tess_straps_csv()),
            maglim=99.0,
            strapsize=0,
            template_path=str(REAL_TEMPLATE_PATH),
            ps1_min_hit_count=5000,
        )
        got = (mask & 16).astype(bool)
        np.testing.assert_array_equal(got, expected)
        self.assertGreater(int(got.sum()), 100_000)


if __name__ == "__main__":
    unittest.main()
