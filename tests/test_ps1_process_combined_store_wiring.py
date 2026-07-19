"""Integration-contract tests for ps1_process <-> combined_store wiring (PR4)."""

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


if __name__ == "__main__":
    unittest.main()
