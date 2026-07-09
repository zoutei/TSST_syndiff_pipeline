"""Unit tests for SEP segmentation helpers in band_utils."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing.band_utils import (
    SepBackgroundResult,
    build_sep_background_segmentation,
    catalog_segment_assignments,
    filled_segment_map,
    remove_background,
)


def _gaussian_image(
    size: int,
    sources: list[tuple[float, float, float, float]],
    background: float = 1.0,
    uncert: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size]
    data = np.full((size, size), background, dtype=np.float32)
    for cx, cy, amp, sigma in sources:
        data += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    uncert_arr = np.full_like(data, uncert)
    return data, uncert_arr


def _catalog_row(
    *,
    pixel_x: float,
    pixel_y: float,
    tess_mag: float,
    source_id: int = 1001,
) -> dict:
    return {
        "source_id": source_id,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "tess_mag": tess_mag,
        "ra": 120.0,
        "dec": 30.0,
        "phot_g_mean_mag": tess_mag,
        "phot_bp_mean_mag": tess_mag + 0.2,
        "phot_rp_mean_mag": tess_mag - 0.2,
    }


class TestBuildSepBackgroundSegmentation:
    def test_detects_injected_sources(self):
        data, uncert = _gaussian_image(
            128,
            [
                (30.0, 30.0, 250.0, 1.2),
                (95.0, 90.0, 220.0, 1.2),
            ],
            background=0.05,
            uncert=0.05,
        )
        result = build_sep_background_segmentation(data, uncert, sigma=3.0, sigma_mask=50)

        assert isinstance(result, SepBackgroundResult)
        assert result.segmap.shape == data.shape
        assert result.mask_bright_stars.shape == data.shape
        assert len(result.objects) >= 2
        segment_ids = np.unique(result.segmap[result.segmap > 0])
        assert len(segment_ids) >= 2


class TestFilledSegmentMap:
    def test_fills_zero_pixels_with_nearest_segment(self):
        segmap = np.zeros((5, 5), dtype=np.int32)
        segmap[1:3, 1:3] = 1
        segmap[3:5, 3:5] = 2

        filled = filled_segment_map(segmap)

        assert filled[0, 2] == 1
        assert filled[2, 0] == 1
        assert filled[2, 4] == 2
        assert filled[4, 2] == 2
        assert filled[1, 1] == 1
        assert filled[3, 3] == 2

        has_id = segmap > 0
        _, indices = ndimage.distance_transform_edt(~has_id, return_indices=True)
        expected = segmap[indices[0], indices[1]]
        np.testing.assert_array_equal(filled, expected)


class TestCatalogSegmentAssignments:
    def test_assigns_catalog_rows_to_segment_ids(self):
        segmap = np.zeros((8, 8), dtype=np.int32)
        segmap[2:5, 2:5] = 1
        segmap[2:5, 5:8] = 2
        filled = filled_segment_map(segmap)
        mask_bright_stars = np.zeros_like(segmap, dtype=bool)

        catalog = pd.DataFrame(
            [
                _catalog_row(pixel_x=3.0, pixel_y=3.0, tess_mag=10.0, source_id=1),
                _catalog_row(pixel_x=6.0, pixel_y=3.0, tess_mag=14.0, source_id=2),
                _catalog_row(pixel_x=0.0, pixel_y=0.0, tess_mag=12.0, source_id=3),
            ]
        )

        assigned = catalog_segment_assignments(
            catalog, filled, mask_bright_stars, segmap=segmap
        )

        assert list(assigned["seg_id_cat"]) == [1, 2, 0]
        assert "seg_id_cat" in assigned.columns
        pd.testing.assert_frame_equal(
            assigned.drop(columns=["seg_id_cat"]),
            catalog,
            check_dtype=False,
        )


class TestRemoveBackgroundRegression:
    def test_background_only_zeros_non_segment_pixels(self):
        data, uncert = _gaussian_image(
            128,
            [
                (30.0, 30.0, 250.0, 1.2),
                (95.0, 90.0, 220.0, 1.2),
            ],
            background=0.05,
            uncert=0.05,
        )
        original = data.copy()

        processed, removed = remove_background(
            data,
            uncert,
            sigma=3.0,
            sigma_mask=50,
            remove_saturated_stars=False,
        )

        assert removed == []
        assert np.any(processed > 0)
        assert np.any(processed == 0)
        assert np.count_nonzero(processed) < np.count_nonzero(original)
        assert processed.shape == original.shape

    def test_catalog_bright_star_removal(self):
        data, uncert = _gaussian_image(
            64,
            [(32.0, 32.0, 120.0, 1.8)],
            background=2.0,
            uncert=0.1,
        )
        catalog = pd.DataFrame([_catalog_row(pixel_x=32.0, pixel_y=32.0, tess_mag=10.0)])

        processed, removed = remove_background(
            data.copy(),
            uncert,
            sigma=2.5,
            sigma_mask=50,
            remove_saturated_stars=True,
            gaia_catalog_pixels=catalog,
            bright_star_mag_threshold=13.0,
        )

        assert len(removed) == 1
        assert removed[0]["removal_reason"] == "catalog_bright_star"
        assert removed[0]["segment_id"] > 0
        assert removed[0]["source_id"] == 1001
        assert processed[32, 32] == 0.0
        assert np.count_nonzero(processed) < np.count_nonzero(data)

    def test_no_catalog_removal_when_star_is_faint(self):
        data, uncert = _gaussian_image(
            64,
            [(32.0, 32.0, 80.0, 1.8)],
            background=2.0,
            uncert=0.1,
        )
        catalog = pd.DataFrame([_catalog_row(pixel_x=32.0, pixel_y=32.0, tess_mag=15.0)])

        processed, removed = remove_background(
            data.copy(),
            uncert,
            sigma=2.5,
            sigma_mask=50,
            remove_saturated_stars=True,
            gaia_catalog_pixels=catalog,
            bright_star_mag_threshold=13.0,
        )

        assert removed == []
        assert processed[32, 32] != 0.0
