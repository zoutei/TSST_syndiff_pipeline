"""Integration-contract tests for the ps1_process <-> combined_store wiring (PR4).

``process_coordinator`` spins up a real ``ProcessPoolExecutor`` and reads from
live thread queues, so a full end-to-end test of it would be slow and, more
importantly, would test machinery (queues/threads/executor orchestration) that
already has zero coverage in this repo and is unrelated to what PR4 actually
changed. What PR4 needs verified is narrower and more valuable: that the
*real* result dict ``process_single_cell`` (run through the *real* SHM
round-trip via ``_materialize_shm_result``, exactly as ``process_coordinator``
receives it) has exactly the shape ``combined_store.publish_combined_cell``
expects, and that publishing it and loading it back reproduces the
background-removed image. This is the actual seam PR4 added, tested for real
end to end minus the multiprocessing plumbing itself.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing.ps1_process import (
    _materialize_shm_result,
    process_single_cell,
)


def _gaussian_image(size: int, cx: float, cy: float, amp: float, sigma: float):
    y, x = np.mgrid[0:size, 0:size]
    data = np.full((size, size), 1.0, dtype=np.float32)
    data += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return data


class RealResultShapeContractTests(unittest.TestCase):
    """Runs the exact function process_coordinator calls, without the executor."""

    def setUp(self) -> None:
        size = 32
        self.combined_image = _gaussian_image(size, 16, 16, amp=50.0, sigma=2.0)
        self.combined_mask = np.zeros((size, size), dtype=np.uint16)
        self.combined_uncert = np.full((size, size), 0.1, dtype=np.float32)
        self.bundle = {
            "skycell_id": "skycell.1234.056",
            "projection": "1234",
            "row_id": 0,
            "x_coord": 0,
            "combined_image": self.combined_image,
            "combined_mask": self.combined_mask,
            "combined_uncert": self.combined_uncert,
            "headers_data": {"r": "SIMPLE=T"},
            "remove_saturated_stars": False,  # simplest path: no Gaia projection needed
        }

    def test_process_single_cell_result_has_expected_shape(self) -> None:
        raw_result = process_single_cell(self.bundle)
        self.assertIsNotNone(raw_result)
        for key in ("skycell_id", "projection", "row_id", "x_coord",
                    "combined_image_shm", "combined_mask_shm", "headers_data", "removed_stars"):
            self.assertIn(key, raw_result)

        result = _materialize_shm_result(raw_result)
        # This is EXACTLY the dict shape process_coordinator's completion
        # handler receives and passes into _publish_combined -> publish_combined_cell.
        for key in ("skycell_id", "projection", "combined_image", "combined_mask",
                    "headers_data", "removed_stars"):
            self.assertIn(key, result)
        self.assertEqual(result["combined_image"].shape, self.combined_image.shape)
        self.assertEqual(result["combined_mask"].shape, self.combined_mask.shape)

    def test_real_result_publishes_and_reloads_through_combined_store(self) -> None:
        """The exact call combined_store.py's _publish_combined closure makes."""
        raw_result = process_single_cell(self.bundle)
        result = _materialize_shm_result(raw_result)

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            recipe = cs.combined_recipe(
                enable_saturation_correction=False,
                remove_saturated_stars=False,
                bright_star_mag_threshold=13.0,
                gaia_version="none",
            )
            ok = cs.publish_combined_cell(
                data_root,
                result["projection"],
                result["skycell_id"],
                recipe,
                combined_image=result["combined_image"],
                combined_mask=result["combined_mask"],
                headers_data=result.get("headers_data"),
                removed_stars=result.get("removed_stars"),
                produced_by="ps1_process",
            )
            self.assertTrue(ok)

            loaded = cs.try_load_combined_cell(
                data_root, result["projection"], result["skycell_id"], recipe
            )
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(
                loaded["combined_image"], result["combined_image"]
            )
            np.testing.assert_array_equal(
                loaded["combined_mask"], result["combined_mask"]
            )
            self.assertEqual(loaded["headers_data"], result["headers_data"])


if __name__ == "__main__":
    unittest.main()
