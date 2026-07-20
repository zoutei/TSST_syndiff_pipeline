"""Integration-contract tests for ps1_process <-> combined_store wiring (PR4)."""

from __future__ import annotations

import json
import queue as _thread_queue
import tempfile
import unittest
from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing.ps1_process import (
    _materialize_shm_result,
    process_coordinator,
    process_single_cell,
)


def _gaussian_image(size: int, cx: float, cy: float, amp: float, sigma: float):
    y, x = np.mgrid[0:size, 0:size]
    data = np.full((size, size), 1.0, dtype=np.float32)
    data += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return data


class RealResultShapeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        size = 32
        self.combined_image = _gaussian_image(size, 16, 16, amp=50.0, sigma=2.0)
        self.combined_mask = np.zeros((size, size), dtype=np.uint16)
        self.combined_uncert = np.full((size, size), 0.1, dtype=np.float32)
        self.bundle = {
            "skycell_id": "skycell.1234.056",
            "projection": "skycell.1234",
            "row_id": 0,
            "x_coord": 0,
            "combined_image": self.combined_image,
            "combined_mask": self.combined_mask,
            "combined_uncert": self.combined_uncert,
            "headers_data": {"r": "SIMPLE=T"},
            "remove_saturated_stars": False,
        }

    def test_process_single_cell_result_has_expected_shape(self) -> None:
        raw_result = process_single_cell(self.bundle)
        self.assertIsNotNone(raw_result)
        result = _materialize_shm_result(raw_result)
        for key in (
            "skycell_id",
            "projection",
            "combined_image",
            "combined_mask",
            "headers_data",
            "removed_stars",
        ):
            self.assertIn(key, result)

    def test_real_result_publishes_and_reloads_through_combined_store(self) -> None:
        raw_result = process_single_cell(self.bundle)
        result = _materialize_shm_result(raw_result)
        parsed = cs._projection_and_cell(result["skycell_id"])
        self.assertIsNotNone(parsed)
        projection, cell = parsed

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            recipe = cs.combined_recipe(gaia_version="none")
            info = cs.publish_combined_cell(
                data_root,
                projection,
                cell,
                combined_image=result["combined_image"],
                combined_mask=result["combined_mask"],
                headers_data=result.get("headers_data"),
                removed_stars=result.get("removed_stars"),
                recipe=recipe,
                producer="ps1_process",
            )
            self.assertIsNotNone(info)
            loaded = cs.try_load_combined_cell(
                data_root, projection, cell, info["fingerprint"]
            )
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded["combined_image"], result["combined_image"])


class RawSkycellInputFingerprintWiringTests(unittest.TestCase):
    """Proves the bug fix end-to-end through ps1_process.py's real call sites.

    Before the fix, ``publish_combined_cell`` was called from
    ``process_coordinator``'s ``_publish_combined`` closure with no
    ``input_fingerprints``, and ``seed_band_cache_from_combined_store``
    called ``combined_fingerprint(...)`` the same way -- both silently
    defaulting to ``()``. A re-downloaded raw skycell (different on-disk
    version token) would then keep resolving to the *same* fingerprint
    forever, so the shared store would serve stale pre-redownload data
    indefinitely. These tests drive the actual production call sites (not a
    reimplementation of them) and check that (a) the fingerprint actually
    used by ``process_coordinator``'s publish path matches
    ``combined_store.raw_skycell_input_fingerprint`` computed the same way
    the seed-lookup side computes it, and (b) a changed raw-skycell version
    token correctly produces a *miss* on the next seed lookup.
    """

    def _write_raw_skycell(self, data_root: Path, projection: str, cell: str, content: bytes) -> None:
        group_dir = (
            data_root / "ps1_skycells_zarr" / "ps1_skycells.zarr" / projection / f"{projection}.{cell}"
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "r.dat").write_bytes(content)

    def test_process_coordinator_publish_uses_raw_skycell_input_fingerprint(self) -> None:
        """The real ``process_coordinator`` -> ``_publish_combined`` call site
        (ps1_process.py) must publish under the same fingerprint that
        ``combined_fingerprint(..., [raw_skycell_input_fingerprint(...)])``
        computes -- i.e. the fix is actually wired in, not just present in
        combined_store.py.
        """
        size = 32
        combined_image = _gaussian_image(size, 16, 16, amp=50.0, sigma=2.0)
        combined_mask = np.zeros((size, size), dtype=np.uint16)
        combined_uncert = np.full((size, size), 0.1, dtype=np.float32)
        projection, cell = "skycell.1234", "056"
        skycell_id = f"{projection}.{cell}"

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self._write_raw_skycell(data_root, projection, cell, b"raw-bytes-v1")

            recipe = cs.combined_recipe(gaia_version="none")
            expected_raw_fp = cs.raw_skycell_input_fingerprint(data_root, projection, cell)
            expected_fp = cs.combined_fingerprint(
                projection, cell, cs.combined_recipe_id(recipe), [expected_raw_fp]
            )

            combined_raw_queue: _thread_queue.Queue = _thread_queue.Queue()
            combined_cell_queue: _thread_queue.Queue = _thread_queue.Queue()
            combined_raw_queue.put(
                {
                    "skycell_id": skycell_id,
                    "projection": projection,
                    "row_id": 0,
                    "x_coord": 0,
                    "task_type": "regular",
                    "combined_image": combined_image,
                    "combined_mask": combined_mask,
                    "combined_uncert": combined_uncert,
                    "headers_data": {"r": "SIMPLE=T"},
                    "remove_saturated_stars": False,
                }
            )
            combined_raw_queue.put(None)

            process_coordinator(
                combined_raw_queue,
                combined_cell_queue,
                cell_buffer={},
                num_workers=1,
                combined_store_data_root=data_root,
                combined_store_recipe=recipe,
            )

            result = combined_cell_queue.get(timeout=5)
            self.assertEqual(result["skycell_id"], skycell_id)

            published_dir = cs.combined_cell_dir(data_root, projection, cell, expected_fp)
            self.assertTrue(
                published_dir.is_dir(),
                f"expected process_coordinator to publish under {expected_fp} at {published_dir}; "
                f"got {sorted(p.name for p in published_dir.parent.iterdir()) if published_dir.parent.exists() else 'no fp dir at all'}",
            )
            sidecar = json.loads((published_dir / "_provenance.json").read_text())
            self.assertEqual(
                sidecar.get("inputs") or sidecar.get("input_fingerprints"),
                [expected_raw_fp],
            )

            # And the seed-lookup side (the other real call site) must
            # independently resolve to that exact same fingerprint.
            hits = cs.seed_band_cache_from_combined_store(data_root, [skycell_id], recipe)
            self.assertIn(skycell_id, hits)
            np.testing.assert_array_equal(hits[skycell_id]["combined_image"], combined_image)

    def test_seed_lookup_misses_after_raw_skycell_redownload(self) -> None:
        """§17 failure-matrix intent: a re-downloaded raw skycell must not
        keep serving the pre-redownload cached combined cell forever.
        """
        size = 32
        combined_image = _gaussian_image(size, 16, 16, amp=50.0, sigma=2.0)
        combined_mask = np.zeros((size, size), dtype=np.uint16)
        headers_data = {"r": "SIMPLE=T"}
        removed_stars: list = []
        projection, cell = "skycell.1234", "056"
        skycell_id = f"{projection}.{cell}"

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self._write_raw_skycell(data_root, projection, cell, b"raw-bytes-v1")

            recipe = cs.combined_recipe(gaia_version="none")
            raw_fp_v1 = cs.raw_skycell_input_fingerprint(data_root, projection, cell)
            info = cs.publish_combined_cell(
                data_root,
                projection,
                cell,
                combined_image=combined_image,
                combined_mask=combined_mask,
                headers_data=headers_data,
                removed_stars=removed_stars,
                recipe=recipe,
                input_fingerprints=[raw_fp_v1],
                producer="ps1_process",
            )
            self.assertIsNotNone(info)

            # Same raw-skycell state -> seed lookup hits.
            hits = cs.seed_band_cache_from_combined_store(data_root, [skycell_id], recipe)
            self.assertIn(skycell_id, hits)

            # Raw skycell re-downloaded: on-disk version token changes.
            self._write_raw_skycell(data_root, projection, cell, b"raw-bytes-v2-different-length")

            hits_after_redownload = cs.seed_band_cache_from_combined_store(
                data_root, [skycell_id], recipe
            )
            self.assertNotIn(
                skycell_id,
                hits_after_redownload,
                "seed lookup must miss (not silently serve stale pre-redownload data) "
                "once the raw skycell's on-disk version token has changed",
            )


if __name__ == "__main__":
    unittest.main()
