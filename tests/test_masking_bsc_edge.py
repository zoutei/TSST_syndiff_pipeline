"""Tests for BSC catalog loading, edge bitmask, and Big_sat BSC merge."""

import unittest

import numpy as np
import pandas as pd

from syndiff_pipeline.common.bsc_catalog import load_bright_star_catalog
from syndiff_pipeline.difference_imaging.stages import masking


class TestBscCatalogLoader(unittest.TestCase):
    def test_first_row_parses(self):
        bsc = load_bright_star_catalog()
        self.assertGreater(len(bsc), 9000)
        row = bsc.iloc[0]
        self.assertEqual(int(row["hr"]), 1)
        self.assertAlmostEqual(float(row["ra"]), 1.29125, places=3)
        self.assertAlmostEqual(float(row["dec"]), 45.229167, places=3)
        self.assertAlmostEqual(float(row["vmag"]), 6.70, places=2)


class TestDetectorEdgeMask(unittest.TestCase):
    def test_left_dead_columns_masked(self):
        crop_bounds = {"x_min": 0, "x_max": 200, "y_min": 100, "y_max": 300}
        edge = masking.detector_edge_mask(
            (200, 200),
            crop_bounds,
            nx=2048,
            ny=2048,
            x_left_dead=44,
            x_right_dead=44,
            y_edge_strip=30,
        )
        self.assertTrue(edge[:, 0].all())
        self.assertTrue(edge[:, 43].all())
        self.assertFalse(edge[:, 44].any())

    def test_interior_crop_no_edge_bits(self):
        crop_bounds = {"x_min": 500, "x_max": 600, "y_min": 400, "y_max": 500}
        edge = masking.detector_edge_mask(
            (100, 100),
            crop_bounds,
            nx=2048,
            ny=2048,
            x_left_dead=44,
            x_right_dead=44,
            y_edge_strip=30,
        )
        self.assertEqual(int(edge.sum()), 0)


class TestBigSatBscMerge(unittest.TestCase):
    def test_bsc_adds_bit2_beyond_gaia_crosses(self):
        image = np.zeros((100, 100), dtype=np.float64)
        gaia_df = pd.DataFrame({"x": [50.0], "y": [50.0], "mag": [10.0]})
        bsc_df = pd.DataFrame({"x": [20.0], "y": [20.0], "vmag": [5.0]})
        mask_gaia = masking.Cat_mask(
            image,
            gaia_df,
            straps_csv="/nonexistent/straps.csv",
            maglim=13.0,
            strapsize=0,
        )
        mask_both = masking.Cat_mask(
            image,
            gaia_df,
            straps_csv="/nonexistent/straps.csv",
            maglim=13.0,
            strapsize=0,
            bsc_df=bsc_df,
        )
        self.assertGreater((mask_both & 2).sum(), (mask_gaia & 2).sum())
        self.assertTrue(mask_both[20, 20] & 2)

    def test_gaia_bit1_unchanged_when_bsc_added(self):
        image = np.zeros((100, 100), dtype=np.float64)
        gaia_df = pd.DataFrame({"x": [50.0], "y": [50.0], "mag": [12.0]})
        mask_gaia = masking.Cat_mask(
            image,
            gaia_df,
            straps_csv="/nonexistent/straps.csv",
            maglim=13.0,
            strapsize=0,
        )
        bsc_df = pd.DataFrame({"x": [80.0], "y": [80.0], "vmag": [6.0]})
        mask_both = masking.Cat_mask(
            image,
            gaia_df,
            straps_csv="/nonexistent/straps.csv",
            maglim=13.0,
            strapsize=0,
            bsc_df=bsc_df,
        )
        self.assertEqual((mask_gaia & 1).sum(), (mask_both & 1).sum())


if __name__ == "__main__":
    unittest.main()
