# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
from __future__ import annotations

import numpy as np
import pandas as pd

from syndiff_pipeline.forward_model.diagnostics import stacked_temporal_residuals as STR


def test_time_blocks_partition_strict_cadences():
    btjd = np.arange(9, dtype=float)
    strict = np.array([True, True, False, True, True, True, False, True, True])
    blocks = STR.time_blocks(btjd, strict)
    joined = np.concatenate(list(blocks.values()))
    assert np.array_equal(np.sort(joined), np.flatnonzero(strict))
    assert not any(2 in row or 6 in row for row in blocks.values())
    assert len(set(joined)) == len(joined)


def test_select_stack_groups_filters_and_assigns_nine_cells():
    n_group, n_time = 11, 6
    metadata = pd.DataFrame({
        "group_index": np.arange(n_group), "slot_index": np.zeros(n_group, dtype=int),
        "star_row": np.arange(n_group), "group_size": [1] * 9 + [2, 1],
        "is_epsf_contributor": [True] * 10 + [False], "tier": np.zeros(n_group, dtype=int),
        "local_row": np.arange(n_group), "support_size": np.full(n_group, 64),
        "x": np.repeat([0., 10., 20.], 3).tolist() + [10., 5.],
        "y": np.tile([0., 10., 20.], 3).tolist() + [5., 5.],
    })
    flux = np.ones((n_group, n_time, 1))
    active = np.ones((n_group, n_time), dtype=bool)
    got = STR.select_stack_groups(metadata, flux, active, np.ones(n_time, dtype=bool))
    # Group 9 (group_size=2, blended) is always excluded; group 10 is a
    # K=1 WCS-only anchor (is_epsf_contributor=False) and is now kept.
    assert set(got.group_index) == set(range(9)) | {10}
    assert set(got.cell) == {f"r{r}c{c}" for r in range(3) for c in range(3)}


def test_select_stack_groups_tags_population():
    metadata = pd.DataFrame({
        "group_index": [0, 1], "slot_index": [0, 0], "star_row": [0, 1],
        "group_size": [1, 1], "is_epsf_contributor": [True, False],
        "tier": [0, 0], "local_row": [0, 1], "support_size": [64, 64],
        "x": [0.0, 10.0], "y": [0.0, 10.0],
    })
    got = STR.select_stack_groups(
        metadata, np.ones((2, 4, 1)), np.ones((2, 4), dtype=bool), np.ones(4, dtype=bool),
    )
    got = got.set_index("group_index")
    assert got.loc[0, "population"] == "epsf_bright"
    assert got.loc[1, "population"] == "wcs_anchor"


def test_catalog_neighbor_pixel_mask_flags_only_pixels_near_a_foreign_source():
    # Star 0 at (100, 100) is the group's own source; star 1 sits 2 px away
    # and should contaminate the pixels closest to it. Star 2 is far away
    # and must not affect anything.
    catalog_x = np.array([100.0, 102.0, 500.0])
    catalog_y = np.array([100.0, 100.0, 500.0])
    pix_x = np.array([99.0, 100.0, 101.0, 102.0, 103.0])
    pix_y = np.full(5, 100.0)
    mask, n_neighbors = STR.catalog_neighbor_pixel_mask(
        pix_x, pix_y, 0, catalog_x, catalog_y,
        exclusion_radius_px=1.0, search_margin_px=6.5,
    )
    assert n_neighbors == 1
    assert list(mask) == [False, False, True, True, True]


def test_catalog_neighbor_pixel_mask_empty_when_isolated():
    catalog_x = np.array([100.0, 500.0])
    catalog_y = np.array([100.0, 500.0])
    pix_x = np.array([99.0, 100.0, 101.0])
    pix_y = np.full(3, 100.0)
    mask, n_neighbors = STR.catalog_neighbor_pixel_mask(
        pix_x, pix_y, 0, catalog_x, catalog_y,
        exclusion_radius_px=1.0, search_margin_px=6.5,
    )
    assert n_neighbors == 0
    assert not mask.any()


def test_select_stack_groups_keeps_multiple_packed_tiers():
    metadata = pd.DataFrame({
        "group_index": [0, 1], "slot_index": [0, 0], "star_row": [0, 1],
        "group_size": [1, 1], "is_epsf_contributor": [True, True],
        "tier": [0, 1], "local_row": [0, 0], "support_size": [64, 128],
        "x": [0.0, 10.0], "y": [0.0, 10.0],
    })
    got = STR.select_stack_groups(
        metadata, np.ones((2, 4, 1)), np.ones((2, 4), dtype=bool), np.ones(4, dtype=bool),
    )
    assert set(got.group_index) == {0, 1}
    assert set(got.tier) == {0, 1}


def test_rasterize_constant_conserves_value_with_subpixel_positions():
    values = np.ones((2, 3))
    coverage = np.ones_like(values)
    image, denom = STR.rasterize_packed_values(
        values, np.array([10.25, 10.75]), np.array([20.1, 19.9]),
        np.array([10., 11., 10.]), np.array([20., 20., 21.]), coverage,
    )
    assert np.allclose(image[denom > 0], 1.0)
    assert np.isclose(denom.sum(), coverage.sum())


def test_robust_group_stack_rejects_single_large_outlier():
    maps = np.stack([np.ones((5, 5)), np.ones((5, 5)) * 1.1, np.ones((5, 5)) * 100])
    stack, count = STR.robust_group_stack(maps, min_groups=2)
    assert np.allclose(stack, 1.05)
    assert np.all(count == 2)


def test_morphology_projection_recognizes_x_dipole():
    n, oversample = 25, 4
    coordinate = (np.arange(n) - (n - 1) / 2) / oversample
    image = np.broadcast_to(coordinate[None, :], (n, n)).copy()
    projection = STR.morphology_projections(image, np.ones((n, n)), oversample=oversample)
    assert projection["x_dipole"] > 0.9
    assert abs(projection["y_dipole"]) < 1e-8
