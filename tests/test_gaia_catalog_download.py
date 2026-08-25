"""Tests for Gaia SCC catalog download helpers in pancakes.py."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing import pancakes


class GaiaCatalogDownloadTests(unittest.TestCase):
    def test_padded_ffi_sky_polygon_is_densely_sampled(self):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [10.0, -5.0]
        wcs.wcs.cdelt = [0.1, 0.1]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        ra, dec = pancakes.padded_ffi_sky_polygon(
            wcs, (10, 20), pixel_padding=10, edge_samples=100
        )
        self.assertEqual(len(ra), 400)
        self.assertEqual(len(dec), 400)
        self.assertTrue(np.all(np.isfinite(ra)))
        self.assertTrue(np.all(np.isfinite(dec)))

    def test_gaia_catalog_columns_include_proper_motion(self):
        cols = pancakes.GAIA_CATALOG_COLUMNS
        for name in (
            "pm",
            "pmra",
            "pmra_error",
            "pmdec",
            "pmdec_error",
        ):
            self.assertIn(name, cols)
        self.assertEqual(len(cols), 15)

    def test_build_gaia_adql_polygon_query_includes_pm_columns(self):
        ra = np.array([10.0, 12.0, 12.0, 10.0])
        dec = np.array([-5.0, -5.0, -3.0, -3.0])
        query = pancakes.build_gaia_adql_polygon_query(ra, dec, magnitude_limit=18.0)
        self.assertIn("pm,", query)
        self.assertIn("pmra,", query)
        self.assertIn("pmdec_error", query)
        self.assertIn("phot_rp_mean_mag < 18.0", query)

    def test_filter_gaia_dataframe_to_polygon_keeps_interior_points(self):
        ra = np.array([0.0, 2.0, 2.0, 0.0])
        dec = np.array([0.0, 0.0, 2.0, 2.0])
        df = pd.DataFrame(
            {
                "ra": [1.0, 5.0],
                "dec": [1.0, 1.0],
            }
        )
        filtered = pancakes.filter_gaia_dataframe_to_polygon(df, ra, dec)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(float(filtered.iloc[0]["ra"]), 1.0)

    def test_filter_gaia_dataframe_to_polygon_handles_ra_wrap(self):
        ra = np.array([359.0, 1.0, 1.0, 359.0])
        dec = np.array([-1.0, -1.0, 1.0, 1.0])
        df = pd.DataFrame({"ra": [0.0, 180.0], "dec": [0.0, 0.0]})

        filtered = pancakes.filter_gaia_dataframe_to_polygon(df, ra, dec)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(float(filtered.iloc[0]["ra"]), 0.0)

    def test_filter_gaia_dataframe_to_polygon_rejects_invalid_coordinates(self):
        ra = np.array([0.0, 2.0, 2.0, 0.0])
        dec = np.array([0.0, 0.0, 2.0, 2.0])
        df = pd.DataFrame({"ra": [1.0, np.nan, 1.0], "dec": [1.0, 1.0, 91.0]})

        filtered = pancakes.filter_gaia_dataframe_to_polygon(df, ra, dec)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(float(filtered.iloc[0]["ra"]), 1.0)

    def test_legacy_catalog_without_metadata_is_not_current(self):
        with TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "gaia.csv"
            catalog.write_text("ra,dec\n1,1\n", encoding="utf-8")
            self.assertFalse(pancakes.gaia_catalog_cache_is_current(str(catalog)))

            pancakes._write_gaia_catalog_metadata(
                str(catalog), pixel_padding=10, magnitude_limit=18.0
            )
            self.assertTrue(pancakes.gaia_catalog_cache_is_current(str(catalog)))

    @patch.object(pancakes, "_download_gaia_catalog_tap")
    @patch.object(pancakes, "_download_gaia_catalog_flathub")
    def test_download_gaia_catalog_tap_backend_skips_flathub(
        self, mock_flathub, mock_tap
    ):
        mock_tap.return_value = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [10.0, -5.0]
        wcs.wcs.cdelt = [0.1, 0.1]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        with patch.object(pancakes, "_save_gaia_catalog_dataframe"):
            pancakes.download_gaia_catalog(
                tess_wcs=wcs,
                data_shape=(10, 10),
                output_path="/tmp/gaia_test_out",
                sector=22,
                camera_id=3,
                ccd_id=3,
                gaia_backend="tap",
            )

        mock_flathub.assert_not_called()
        mock_tap.assert_called_once()

    @patch.object(pancakes, "_download_gaia_catalog_tap")
    @patch.object(pancakes, "_download_gaia_catalog_flathub")
    def test_download_gaia_catalog_auto_falls_back_to_tap(
        self, mock_flathub, mock_tap
    ):
        mock_flathub.side_effect = RuntimeError("flathub down")
        mock_tap.return_value = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [10.0, -5.0]
        wcs.wcs.cdelt = [0.1, 0.1]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        with patch.object(pancakes, "_save_gaia_catalog_dataframe"):
            pancakes.download_gaia_catalog(
                tess_wcs=wcs,
                data_shape=(10, 10),
                output_path="/tmp/gaia_test_out",
                sector=22,
                camera_id=3,
                ccd_id=3,
                gaia_backend="auto",
            )

        mock_flathub.assert_called_once()
        mock_tap.assert_called_once()

    @patch.object(pancakes, "_fetch_flathub_numpy")
    def test_flathub_download_applies_bbox_and_polygon_filter(self, mock_fetch):
        arr = np.array(
            [(1.0, 1.0), (5.0, 1.0)],
            dtype=[("ra", "f8"), ("dec", "f8")],
        )
        mock_fetch.return_value = arr

        ra = np.array([0.0, 2.0, 2.0, 0.0])
        dec = np.array([0.0, 0.0, 2.0, 2.0])
        df = pancakes._download_gaia_catalog_flathub(ra, dec, magnitude_limit=18.0)

        mock_fetch.assert_called_once()
        kwargs = mock_fetch.call_args.kwargs
        self.assertEqual(kwargs["ra"], (0.0, 2.0))
        self.assertEqual(kwargs["dec"], (0.0, 2.0))
        self.assertEqual(kwargs["phot_rp_mean_mag"], (0.0, 18.0))
        self.assertEqual(len(df), 1)
        self.assertEqual(float(df.iloc[0]["ra"]), 1.0)


if __name__ == "__main__":
    unittest.main()
