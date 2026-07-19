"""Tests for the shared canonical convolved-skycell store (Phase 2 data layer, PR5).

This module is intentionally NOT wired into ps1_process.py yet (see
convolved_store.py's docstring for why) — these tests exercise the store in
isolation, mirroring test_combined_store.py's coverage.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.common.provenance.reindex import reindex_shared_store
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path
from syndiff_pipeline.template_creation.processing import convolved_store as vs


def _recipe(**overrides):
    defaults = dict(psf_sigma=60.0, radius=470, mode="constant")
    defaults.update(overrides)
    return vs.convolved_recipe(**defaults)


class RecipeTests(unittest.TestCase):
    def test_recipe_id_changes_with_sigma(self) -> None:
        r1 = _recipe()
        r2 = _recipe(psf_sigma=45.0)
        self.assertNotEqual(r1.recipe_id(), r2.recipe_id())

    def test_recipe_records_canonical_padding_scope(self) -> None:
        r = _recipe()
        self.assertEqual(r.params["padding"], "same_projection_only")


class ConvolvedCellDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_deterministic(self) -> None:
        recipe = _recipe()
        d1 = vs.convolved_cell_dir(self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA")
        d2 = vs.convolved_cell_dir(self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA")
        self.assertEqual(d1, d2)

    def test_different_combined_input_changes_dir(self) -> None:
        recipe = _recipe()
        d1 = vs.convolved_cell_dir(self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA")
        d2 = vs.convolved_cell_dir(self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpB")
        self.assertNotEqual(d1, d2)  # Merkle: a recomputed combined cell invalidates this

    def test_located_under_convolved_path(self) -> None:
        recipe = _recipe()
        d = vs.convolved_cell_dir(self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA")
        self.assertTrue(str(d).startswith(str(ps1_convolved_zarr_path(self.root))))


class LoadMissTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_miss_on_empty_store(self) -> None:
        loaded = vs.try_load_convolved_cell(
            self.root, "1234", "skycell.1234.056", _recipe(), combined_fp="fpA"
        )
        self.assertIsNone(loaded)

    def test_corrupt_file_is_miss_not_raise(self) -> None:
        recipe = _recipe()
        cell_dir = vs.convolved_cell_dir(
            self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA"
        )
        cell_dir.mkdir(parents=True)
        (cell_dir / "arrays.npz").write_bytes(b"garbage")
        loaded = vs.try_load_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA"
        )
        self.assertIsNone(loaded)


class PublishLoadRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.image = np.arange(20, dtype=np.float32).reshape(4, 5)
        self.headers = {"r": "SIMPLE=T"}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip(self) -> None:
        recipe = _recipe()
        ok = vs.publish_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_fp="fpA", convolved_image=self.image,
            headers_data=self.headers, produced_by="run-x",
        )
        self.assertTrue(ok)
        loaded = vs.try_load_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA"
        )
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded["convolved_image"], self.image)
        self.assertEqual(loaded["headers_data"], self.headers)

    def test_no_leftover_tmp_dir(self) -> None:
        recipe = _recipe()
        vs.publish_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_fp="fpA", convolved_image=self.image,
        )
        cell_dir = vs.convolved_cell_dir(
            self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA"
        )
        self.assertEqual(list(cell_dir.parent.iterdir()), [cell_dir])

    def test_second_publish_is_noop_true(self) -> None:
        recipe = _recipe()
        vs.publish_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_fp="fpA", convolved_image=self.image,
        )
        ok2 = vs.publish_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_fp="fpA", convolved_image=self.image,
        )
        self.assertTrue(ok2)

    def test_publish_failure_is_non_fatal(self) -> None:
        recipe = _recipe()

        class Unserializable:
            def __array__(self):
                raise RuntimeError("boom")

        ok = vs.publish_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_fp="fpA", convolved_image=Unserializable(),
        )
        self.assertFalse(ok)
        self.assertIsNone(
            vs.try_load_convolved_cell(
                self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA"
            )
        )


class ReindexIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_published_cell_reindexes_as_complete(self) -> None:
        recipe = _recipe()
        vs.publish_convolved_cell(
            self.root, "1234", "skycell.1234.056", recipe,
            combined_fp="fpA", convolved_image=np.ones((2, 2), dtype=np.float32),
            produced_by="run-x",
        )
        store = ProvenanceStore(self.root / "provenance.db")
        counts = reindex_shared_store(
            store, ps1_convolved_zarr_path(self.root), model.CONVOLVED_SKYCELL
        )
        self.assertEqual(counts, {"sidecar": 1, "legacy": 0})

        cell_dir = vs.convolved_cell_dir(
            self.root, "1234", "skycell.1234.056", recipe, combined_fp="fpA"
        )
        art = store.get_artifact(cell_dir.name)
        self.assertIsNotNone(art)
        self.assertEqual(art["state"], "complete")
        self.assertEqual(store.artifact_inputs(cell_dir.name), ["fpA"])


if __name__ == "__main__":
    unittest.main()
