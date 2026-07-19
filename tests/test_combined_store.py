"""Tests for the shared combined-skycell store (Phase 1, PR4)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.common.provenance.reindex import reindex_shared_store
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import ps1_combined_zarr_path
from syndiff_pipeline.template_creation.processing import combined_store as cs


def _recipe(**overrides):
    defaults = dict(
        enable_saturation_correction=True,
        remove_saturated_stars=False,
        bright_star_mag_threshold=13.0,
        gaia_version="none",
    )
    defaults.update(overrides)
    return cs.combined_recipe(**defaults)


class ProjectionParsingTests(unittest.TestCase):
    def test_parses_standard_name(self) -> None:
        self.assertEqual(cs.projection_from_skycell_name("skycell.1234.056"), "1234")

    def test_returns_none_on_bad_name(self) -> None:
        self.assertIsNone(cs.projection_from_skycell_name("bogus"))
        self.assertIsNone(cs.projection_from_skycell_name(None))


class GaiaVersionStampTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_none_path_is_stable_sentinel(self) -> None:
        self.assertEqual(cs.gaia_version_stamp(None), "none")
        self.assertEqual(cs.gaia_version_stamp(""), "none")

    def test_missing_file_is_unknown_not_raise(self) -> None:
        stamp = cs.gaia_version_stamp(str(self.root / "nope.csv"))
        self.assertIn("unknown", stamp)

    def test_real_file_stamp_changes_with_mtime(self) -> None:
        f = self.root / "gaia.csv"
        f.write_text("a,b\n1,2\n")
        stamp1 = cs.gaia_version_stamp(str(f))
        time.sleep(0.01)
        f.write_text("a,b\n1,2,3\n")  # different size AND mtime
        stamp2 = cs.gaia_version_stamp(str(f))
        self.assertNotEqual(stamp1, stamp2)


class RecipeAndFingerprintTests(unittest.TestCase):
    def test_recipe_id_changes_with_params(self) -> None:
        r1 = _recipe()
        r2 = _recipe(bright_star_mag_threshold=12.0)
        self.assertNotEqual(r1.recipe_id(), r2.recipe_id())

    def test_raw_skycell_fingerprint_differs_by_skycell(self) -> None:
        fp1 = cs.raw_skycell_input_fingerprint("1234", "056")
        fp2 = cs.raw_skycell_input_fingerprint("1234", "057")
        self.assertNotEqual(fp1, fp2)

    def test_raw_skycell_fingerprint_deterministic(self) -> None:
        fp1 = cs.raw_skycell_input_fingerprint("1234", "056")
        fp2 = cs.raw_skycell_input_fingerprint("1234", "056")
        self.assertEqual(fp1, fp2)


class CombinedCellDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_deterministic_across_calls(self) -> None:
        recipe = _recipe()
        d1 = cs.combined_cell_dir(self.root, "1234", "skycell.1234.056", recipe)
        d2 = cs.combined_cell_dir(self.root, "1234", "skycell.1234.056", recipe)
        self.assertEqual(d1, d2)

    def test_different_recipe_different_dir(self) -> None:
        d1 = cs.combined_cell_dir(self.root, "1234", "skycell.1234.056", _recipe())
        d2 = cs.combined_cell_dir(
            self.root, "1234", "skycell.1234.056", _recipe(remove_saturated_stars=True)
        )
        self.assertNotEqual(d1, d2)

    def test_located_under_combined_zarr_path(self) -> None:
        recipe = _recipe()
        d = cs.combined_cell_dir(self.root, "1234", "skycell.1234.056", recipe)
        self.assertTrue(str(d).startswith(str(ps1_combined_zarr_path(self.root))))


class LoadMissTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_miss_on_empty_store(self) -> None:
        loaded = cs.try_load_combined_cell(self.root, "1234", "skycell.1234.056", _recipe())
        self.assertIsNone(loaded)

    def test_corrupt_arrays_file_is_miss_not_raise(self) -> None:
        recipe = _recipe()
        cell_dir = cs.combined_cell_dir(self.root, "1234", "skycell.1234.056", recipe)
        cell_dir.mkdir(parents=True)
        (cell_dir / "arrays.npz").write_bytes(b"not a valid npz file")
        loaded = cs.try_load_combined_cell(self.root, "1234", "skycell.1234.056", recipe)
        self.assertIsNone(loaded)


class PublishLoadRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.image = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.mask = np.zeros((3, 4), dtype=np.uint16)
        self.mask[1, 1] = 5
        self.headers = {"r": "SIMPLE=T"}
        self.removed_stars = [{"x": 1.0, "y": 2.0, "mag": 11.0}]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip(self) -> None:
        recipe = _recipe()
        ok = cs.publish_combined_cell(
            self.root,
            "1234",
            "skycell.1234.056",
            recipe,
            combined_image=self.image,
            combined_mask=self.mask,
            headers_data=self.headers,
            removed_stars=self.removed_stars,
            produced_by="run-x",
        )
        self.assertTrue(ok)
        loaded = cs.try_load_combined_cell(self.root, "1234", "skycell.1234.056", recipe)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded["combined_image"], self.image)
        np.testing.assert_array_equal(loaded["combined_mask"], self.mask)
        self.assertEqual(loaded["headers_data"], self.headers)
        self.assertEqual(loaded["removed_stars"], self.removed_stars)

    def test_no_leftover_tmp_dir(self) -> None:
        recipe = _recipe()
        cs.publish_combined_cell(
            self.root,
            "1234",
            "skycell.1234.056",
            recipe,
            combined_image=self.image,
            combined_mask=self.mask,
        )
        cell_dir = cs.combined_cell_dir(self.root, "1234", "skycell.1234.056", recipe)
        siblings = list(cell_dir.parent.iterdir())
        self.assertEqual(siblings, [cell_dir])  # only the final dir, no _tmp_* leftover

    def test_second_publish_is_noop_true(self) -> None:
        recipe = _recipe()
        cs.publish_combined_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_image=self.image, combined_mask=self.mask,
        )
        # Concurrent/duplicate publish of the identical fingerprint: must not
        # raise or corrupt the existing artifact, and must report success.
        ok2 = cs.publish_combined_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_image=self.image, combined_mask=self.mask,
        )
        self.assertTrue(ok2)

    def test_publish_failure_is_non_fatal(self) -> None:
        recipe = _recipe()
        # Force a failure by making the destination's parent unwritable-ish:
        # pass a combined_image that cannot be coerced, to exercise the guard.
        class Unserializable:
            def __array__(self):
                raise RuntimeError("boom")

        ok = cs.publish_combined_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_image=Unserializable(), combined_mask=self.mask,
        )
        self.assertFalse(ok)
        # And nothing should be readable afterward.
        self.assertIsNone(
            cs.try_load_combined_cell(self.root, "1234", "skycell.1234.056", recipe)
        )


class SeedBandCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.recipe = _recipe()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mixed_hits_and_misses(self) -> None:
        image = np.ones((2, 2), dtype=np.float32)
        mask = np.zeros((2, 2), dtype=np.uint16)
        cs.publish_combined_cell(
            self.root, "1234", "skycell.1234.056", self.recipe,
            combined_image=image, combined_mask=mask,
        )
        hits = cs.seed_band_cache_from_combined_store(
            self.root,
            ["skycell.1234.056", "skycell.1234.057", "bogus-name"],
            self.recipe,
        )
        self.assertEqual(set(hits.keys()), {"skycell.1234.056"})
        np.testing.assert_array_equal(hits["skycell.1234.056"]["combined_image"], image)

    def test_empty_input(self) -> None:
        self.assertEqual(cs.seed_band_cache_from_combined_store(self.root, [], self.recipe), {})


class ReindexIntegrationTests(unittest.TestCase):
    """Proves combined_store's on-disk layout is exactly what PR1's reindex expects."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_published_cell_reindexes_as_complete(self) -> None:
        recipe = _recipe()
        image = np.ones((2, 2), dtype=np.float32)
        mask = np.zeros((2, 2), dtype=np.uint16)
        cs.publish_combined_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_image=image, combined_mask=mask, produced_by="run-x",
        )
        store = ProvenanceStore(self.root / "provenance.db")
        counts = reindex_shared_store(
            store, ps1_combined_zarr_path(self.root), model.COMBINED_SKYCELL
        )
        self.assertEqual(counts, {"sidecar": 1, "legacy": 0})

        cell_dir = cs.combined_cell_dir(self.root, "1234", "skycell.1234.056", recipe)
        fp = cell_dir.name
        art = store.get_artifact(fp)
        self.assertIsNotNone(art)
        self.assertEqual(art["state"], "complete")
        self.assertEqual(art["kind"], model.COMBINED_SKYCELL)


if __name__ == "__main__":
    unittest.main()
