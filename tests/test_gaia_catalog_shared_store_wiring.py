"""Coverage for the Gaia-catalog wiring (2026-08-21; rewritten for the v2
seam-neighbour fix, doc/seam_neighbour_fix_plan_20260930.md):

1. ps1_process.run_modern_sliding_window_pipeline must abort (return an
   ``{"error": ...}`` dict, not silently degrade to ``catalog=None``) when a
   Gaia catalogue is required (remove_saturated_stars/enable_saturation_
   correction) but the per-projection catalogues cannot be built/loaded
   (``gaia_projection_catalog.ensure_projection_catalog`` /
   ``load_catalog_for_projections``).
2. combined_store.production_combined_recipe (schema v2) stamps
   ``gaia_version`` with the SCC-independent projection scheme, never with the
   publishing SCC's own catalogue; the catalogue content enters the combined
   fingerprint as a per-cell input, and an absent catalogue makes it undefined.
3. cross_projection_padding's live cache-miss fallback
   (_load_padding_source_once) must select removal rows through
   ``ps1_process.select_removal_catalog`` and fail loudly
   (``MissingRemovalCatalogError``) when no catalogue is available.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import cross_projection_padding as cpp
from syndiff_pipeline.template_creation.processing import gaia_projection_catalog as gpc
from tests.seam_helpers import write_projection_catalog


class TestCatalogLoadFailureAborts(unittest.TestCase):
    def _run(self, tmp: str, *, catalog_path: str | None = None):
        import syndiff_pipeline.template_creation.processing.ps1_process as ps1p

        # v2: the run builds per-projection catalogues via
        # ensure_projection_catalog; make that fail (no network in tests).
        with (
            mock.patch.object(
                gpc, "ensure_projection_catalog", side_effect=RuntimeError("download failed"),
            ),
            mock.patch.object(ps1p, "get_projections_from_csv", return_value=["skycell.1234"]),
            mock.patch.object(ps1p, "load_csv_data", return_value=pd.DataFrame({"projection": ["1234"], "y": [0]})),
            mock.patch.object(ps1p, "expected_convolved_skycells", return_value=set()),
            # v2 loads the catalogues after the zarr is opened (padding sources
            # are needed first), so stand the zarr open in.
            mock.patch.object(ps1p, "zarr"),
        ):
            csv_path = os.path.join(tmp, "master_skycells.csv")
            Path(csv_path).write_text("NAME\nskycell.1234.001\n")
            return ps1p.run_modern_sliding_window_pipeline(
                sector=20, camera=3, ccd=3,
                data_root=tmp,
                mapping_csv_path=csv_path,
                remove_saturated_stars=True,
            )

    def test_missing_catalog_file_aborts_with_error(self):
        with self._make_tmp() as tmp:
            result = self._run(tmp)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("Gaia catalog", result["error"])

    def _make_tmp(self):
        import tempfile

        return tempfile.TemporaryDirectory()

    def test_run_does_not_proceed_past_catalog_failure(self):
        """A catalog failure must short-circuit before any worker thread starts.
        (v2: the zarr handle is opened first, since padding-source projections
        must be known before the catalogues are built; no workers by then.)"""
        import syndiff_pipeline.template_creation.processing.ps1_process as ps1p

        with self._make_tmp() as tmp:
            with mock.patch.object(ps1p, "threading") as threading_mock:
                result = self._run(tmp)
                threading_mock.Thread.assert_not_called()
        self.assertIn("error", result)


class TestGaiaVersionStampResolution(unittest.TestCase):
    """Schema v2: the recipe names the catalogue *scheme*; the catalogue
    *content* is a per-cell input fingerprint (the old "loaded"/"none" stamp
    resolved from the SCC's own catalogue file is gone by design)."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.data_root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_scheme_stamped_without_scc_identity(self):
        recipe = cs.production_combined_recipe({"remove_saturated_stars": True})
        self.assertEqual(recipe["gaia_version"], gpc.GAIA_PROJECTION_SCHEME)
        self.assertEqual(recipe["bright_star_removal"], "footprint_v1")

    def test_flags_off_stamps_none(self):
        recipe = cs.production_combined_recipe(
            {"remove_saturated_stars": False, "enable_saturation_correction": False}
        )
        self.assertEqual(recipe["gaia_version"], "none")
        self.assertEqual(recipe["bright_star_removal"], "none")

    def test_recipe_independent_of_scc_identity_catalog_path_and_files(self):
        """The recipe no longer depends on data_root/sector/camera/ccd,
        catalog_path, or any per-SCC catalogue file on disk."""
        base = cs.production_combined_recipe({"remove_saturated_stars": True})
        write_projection_catalog(self.data_root, "1234")
        with_identity = cs.production_combined_recipe(
            {"remove_saturated_stars": True},
            data_root=self.data_root, sector=20, camera=3, ccd=3,
        )
        other_scc = cs.production_combined_recipe(
            {"remove_saturated_stars": True},
            data_root=self.data_root, sector=51, camera=4, ccd=4,
        )
        # v2: an explicit (even missing) catalog_path used to flip the stamp to "none".
        explicit_missing = cs.production_combined_recipe(
            {"remove_saturated_stars": True,
             "catalog_path": str(Path(self.data_root) / "does_not_exist.csv")},
            data_root=self.data_root, sector=20, camera=3, ccd=3,
        )
        self.assertEqual(base, with_identity)
        self.assertEqual(base, other_scc)
        self.assertEqual(base, explicit_missing)

    def test_combined_fingerprint_tracks_projection_catalog_content(self):
        recipe = cs.production_combined_recipe({"remove_saturated_stars": True})
        proj, cell = "skycell.1234", "001"
        # No catalogue file for the projection: fingerprint undefined -> missing.
        self.assertIsNone(cs.expected_combined_fingerprint(self.data_root, proj, cell, recipe))
        self.assertIsNone(cs.resolve_combined_fingerprint_for_recipe(self.data_root, proj, cell, recipe))

        write_projection_catalog(self.data_root, "1234", n=3)
        fp_a = cs.expected_combined_fingerprint(self.data_root, proj, cell, recipe)
        self.assertIsNotNone(fp_a)
        write_projection_catalog(self.data_root, "1234", n=4)  # catalogue content changed
        fp_b = cs.expected_combined_fingerprint(self.data_root, proj, cell, recipe)
        self.assertNotEqual(fp_a, fp_b)

        # A catalogue-free recipe never depends on the catalogue.
        none_recipe = cs.production_combined_recipe({"remove_saturated_stars": False})
        self.assertIsNotNone(cs.expected_combined_fingerprint(self.data_root, "skycell.9999", cell, none_recipe))


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
        from syndiff_pipeline.template_creation.processing.band_utils import REMOVAL_CONVENTION

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
                # v2: selection goes through select_removal_catalog, not project_gaia_to_skycell.
                "syndiff_pipeline.template_creation.processing.ps1_process.select_removal_catalog",
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
        self.assertEqual(kwargs["convention"], REMOVAL_CONVENTION)
        _, sel_kwargs = project_mock.call_args
        self.assertEqual(sel_kwargs["bright_star_mag_threshold"], 11.0)
        self.assertEqual(sel_kwargs["skycell_id"], "skycell.9999.001")

    def test_load_padding_source_once_raises_without_catalog(self):
        # v2: a missing catalogue with removal on is fatal (MissingRemovalCatalogError),
        # not a silent "skip removal" (the old behaviour passed gaia_catalog_pixels=None).
        from syndiff_pipeline.template_creation.processing.ps1_process import MissingRemovalCatalogError

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
            mock.patch.object(cpp, "remove_background", return_value=(data, [])) as remove_mock,
        ):
            with self.assertRaises(MissingRemovalCatalogError):
                cpp._load_padding_source_once(
                    placement,
                    band_cache=None,
                    ingest_config={},
                    remove_saturated_stars=True,
                    current_df=pd.DataFrame(),
                    gaia_catalog=None,
                )
        remove_mock.assert_not_called()

    def test_dead_padding_job_path_was_removed(self):
        """Regression guard: the unreachable PaddingJob/_process_padding_job
        cluster (zero callers, never wired for catalog removal) was deleted
        rather than patched -- assert it stays gone."""
        for name in ("PaddingJob", "analyze_padding_jobs", "_process_padding_job"):
            self.assertFalse(hasattr(cpp, name), f"{name} should have been removed")


if __name__ == "__main__":
    unittest.main()
