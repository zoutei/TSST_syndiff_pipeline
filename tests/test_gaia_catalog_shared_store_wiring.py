"""Coverage for the Gaia-catalog wiring fix (2026-08-21):

1. ps1_process.run_modern_sliding_window_pipeline must abort (return an
   ``{"error": ...}`` dict, not silently degrade to ``catalog=None``) when a
   Gaia catalog is required (remove_saturated_stars/enable_saturation_
   correction) but fails to load.
2. combined_store.production_combined_recipe must stamp gaia_version from
   the actually-resolved default catalog path (not just an explicit
   catalog_path override) when SCC identity is supplied, and fall back to
   the old "none"-when-unset behavior when it isn't.
3. cross_projection_padding's live cache-miss fallback
   (_load_padding_source_once) must apply catalog-based bright-star removal
   when given a Gaia catalog, not just the magnitude-blind PS1-mask pass.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from syndiff_pipeline.common.scc_paths import default_gaia_catalog_path
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import cross_projection_padding as cpp


class TestCatalogLoadFailureAborts(unittest.TestCase):
    def _run(self, tmp: str, *, catalog_path: str):
        import syndiff_pipeline.template_creation.processing.ps1_process as ps1p

        with (
            mock.patch.object(ps1p, "get_projections_from_csv", return_value=["skycell.1234"]),
            mock.patch.object(ps1p, "load_csv_data", return_value=pd.DataFrame({"projection": ["1234"], "y": [0]})),
            mock.patch.object(ps1p, "expected_convolved_skycells", return_value=set()),
        ):
            csv_path = os.path.join(tmp, "master_skycells.csv")
            Path(csv_path).write_text("NAME\nskycell.1234.001\n")
            return ps1p.run_modern_sliding_window_pipeline(
                sector=20, camera=3, ccd=3,
                data_root=tmp,
                mapping_csv_path=csv_path,
                remove_saturated_stars=True,
                catalog_path=catalog_path,
            )

    def test_missing_catalog_file_aborts_with_error(self):
        with self._make_tmp() as tmp:
            result = self._run(tmp, catalog_path=os.path.join(tmp, "does_not_exist.csv"))
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("Gaia catalog", result["error"])

    def _make_tmp(self):
        import tempfile

        return tempfile.TemporaryDirectory()

    def test_run_does_not_proceed_past_catalog_failure(self):
        """A catalog failure must short-circuit before any zarr/ingest work starts."""
        import syndiff_pipeline.template_creation.processing.ps1_process as ps1p

        with self._make_tmp() as tmp:
            with mock.patch.object(ps1p, "zarr") as zarr_mock:
                result = self._run(tmp, catalog_path=os.path.join(tmp, "missing.csv"))
                zarr_mock.open.assert_not_called()
        self.assertIn("error", result)


class TestGaiaVersionStampResolution(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.data_root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write_catalog(self):
        p = default_gaia_catalog_path(self.data_root, 20, 3, 3)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ra,dec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag\n1.0,2.0,10,10,10\n")
        return p

    def test_unresolved_catalog_path_stamps_none_without_scc_identity(self):
        recipe = cs.production_combined_recipe({"remove_saturated_stars": True})
        self.assertEqual(recipe["gaia_version"], "none")

    def test_resolved_default_catalog_path_stamps_real_identity_with_scc_identity(self):
        self._write_catalog()
        recipe = cs.production_combined_recipe(
            {"remove_saturated_stars": True},
            data_root=self.data_root, sector=20, camera=3, ccd=3,
        )
        self.assertNotEqual(recipe["gaia_version"], "none")
        self.assertIn(str(default_gaia_catalog_path(self.data_root, 20, 3, 3)), recipe["gaia_version"])

    def test_stamp_changes_when_catalog_content_changes(self):
        p = self._write_catalog()
        recipe_a = cs.production_combined_recipe(
            {"remove_saturated_stars": True},
            data_root=self.data_root, sector=20, camera=3, ccd=3,
        )
        # Simulate a re-download / updated catalog: different size + mtime.
        p.write_text(
            "ra,dec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag\n1.0,2.0,10,10,10\n3.0,4.0,11,11,11\n"
        )
        os.utime(p, None)
        recipe_b = cs.production_combined_recipe(
            {"remove_saturated_stars": True},
            data_root=self.data_root, sector=20, camera=3, ccd=3,
        )
        self.assertNotEqual(recipe_a["gaia_version"], recipe_b["gaia_version"])
        self.assertNotEqual(recipe_a, recipe_b)

    def test_explicit_catalog_path_override_still_wins(self):
        explicit = Path(self.data_root) / "custom.csv"
        explicit.write_text("ra,dec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag\n1.0,2.0,10,10,10\n")
        recipe = cs.production_combined_recipe(
            {"remove_saturated_stars": True, "catalog_path": str(explicit)},
            data_root=self.data_root, sector=20, camera=3, ccd=3,
        )
        self.assertIn(str(explicit), recipe["gaia_version"])


class TestCrossProjectionPaddingCatalogThreading(unittest.TestCase):
    def test_load_padding_source_once_applies_catalog_based_removal_on_cache_miss(self):
        placement = cpp.PaddingPlacement(
            source_skycell="skycell.9999.001",
            source_projection="skycell.9999",
            recipient_skycell="skycell.1234.001",
            location="left",
            recipient_index=0,
            priority=0,
        )
        data = np.ones((8, 8), dtype=np.float32)
        mask = np.zeros((8, 8), dtype=np.int32)
        uncert = np.ones((8, 8), dtype=np.float32)
        catalog = pd.DataFrame({"ra": [1.0], "dec": [2.0], "phot_g_mean_mag": [10.0]})
        sentinel_pixels = pd.DataFrame({"pixel_x": [4], "pixel_y": [4], "tess_mag": [9.0]})

        with (
            mock.patch.object(
                cpp, "create_cell_wcs", return_value=mock.MagicMock(),
            ),
            mock.patch(
                "syndiff_pipeline.template_creation.processing.ps1_process._load_skycell_raw_bands",
                return_value=(["r"], ["r"], {"r": 1.0}, {"r": "h"}, {"r": "h"}),
            ),
            mock.patch.object(
                cpp, "process_skycell_bands", return_value=(data, mask, uncert),
            ),
            mock.patch(
                "syndiff_pipeline.template_creation.processing.ps1_process.project_gaia_to_skycell",
                return_value=sentinel_pixels,
            ) as project_mock,
            mock.patch.object(
                cpp, "remove_background", return_value=(data, []),
            ) as remove_mock,
        ):
            cpp._load_padding_source_once(
                placement,
                band_cache=None,
                ingest_config={},
                remove_saturated_stars=True,
                current_df=pd.DataFrame(),
                gaia_catalog=catalog,
                bright_star_mag_threshold=11.0,
            )

        project_mock.assert_called_once()
        remove_mock.assert_called_once()
        _, kwargs = remove_mock.call_args
        self.assertIs(kwargs["gaia_catalog_pixels"], sentinel_pixels)
        self.assertEqual(kwargs["bright_star_mag_threshold"], 11.0)

    def test_load_padding_source_once_skips_projection_without_catalog(self):
        placement = cpp.PaddingPlacement(
            source_skycell="skycell.9999.001",
            source_projection="skycell.9999",
            recipient_skycell="skycell.1234.001",
            location="left",
            recipient_index=0,
            priority=0,
        )
        data = np.ones((8, 8), dtype=np.float32)
        mask = np.zeros((8, 8), dtype=np.int32)
        uncert = np.ones((8, 8), dtype=np.float32)

        with (
            mock.patch.object(cpp, "create_cell_wcs", return_value=mock.MagicMock()),
            mock.patch(
                "syndiff_pipeline.template_creation.processing.ps1_process._load_skycell_raw_bands",
                return_value=(["r"], ["r"], {"r": 1.0}, {"r": "h"}, {"r": "h"}),
            ),
            mock.patch.object(cpp, "process_skycell_bands", return_value=(data, mask, uncert)),
            mock.patch(
                "syndiff_pipeline.template_creation.processing.ps1_process.project_gaia_to_skycell",
            ) as project_mock,
            mock.patch.object(cpp, "remove_background", return_value=(data, [])) as remove_mock,
        ):
            cpp._load_padding_source_once(
                placement,
                band_cache=None,
                ingest_config={},
                remove_saturated_stars=True,
                current_df=pd.DataFrame(),
                gaia_catalog=None,
            )

        project_mock.assert_not_called()
        _, kwargs = remove_mock.call_args
        self.assertIsNone(kwargs["gaia_catalog_pixels"])

    def test_dead_padding_job_path_was_removed(self):
        """Regression guard: the unreachable PaddingJob/_process_padding_job
        cluster (zero callers, never wired for catalog removal) was deleted
        rather than patched -- assert it stays gone."""
        for name in ("PaddingJob", "analyze_padding_jobs", "_process_padding_job"):
            self.assertFalse(hasattr(cpp, name), f"{name} should have been removed")


if __name__ == "__main__":
    unittest.main()
