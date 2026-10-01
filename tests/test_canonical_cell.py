"""Row-path canonical snapshot == the single reference definition (canonical_cell).

Covers the producer defects of doc/seam_neighbour_fix_plan_20260930.md:
A (cross-projection patches leaking into the next row's snapshot), B (last-row
top strip blanked), and the per-row x anchor (rows starting at different x).
Synthetic cells use the production geometry constants (PAD = OVERLAP = 480),
so they are 1000 px; a small blur keeps the test fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import canonical_cell as cc
from syndiff_pipeline.template_creation.processing import ps1_process as pp
from syndiff_pipeline.template_creation.processing import convolution_utils

W = H = 1000
SIGMA, RADIUS = 5.0, 20

# Ragged rows: row 0 starts at x=1, row 1 at x=0, row 2 at x=1 (different first x),
# like a mapping list cut by a CCD footprint.
ROWS = {
    0: [("skycell.9999.001", 1), ("skycell.9999.002", 2)],
    1: [("skycell.9999.010", 0), ("skycell.9999.011", 1), ("skycell.9999.012", 2)],
    2: [("skycell.9999.021", 1)],
}


def _metadata():
    names = [n for cells in ROWS.values() for n, _ in cells]
    xs = [x for cells in ROWS.values() for _, x in cells]
    return {
        "projection": "9999",
        "rows": {r: list(c) for r, c in ROWS.items()},
        "cell_width": W,
        "cell_height": H,
        "max_cells_per_row": max(len(c) for c in ROWS.values()),
        "span_cells": max(xs) - min(xs) + 1,
        "starting_x": min(xs),
        "cell_dimensions": {n: (W, H) for n in names},
    }


def _images():
    rng = np.random.default_rng(7)
    out = {}
    for cells in ROWS.values():
        for name, _ in cells:
            img = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
            img[0, :] = np.nan  # PS1-masked border
            img[:, -1] = np.nan
            out[name] = img
    return out


def _run_row_path(metadata, images, junk_cross_projection=True):
    """Drive the production row-path primitives exactly as process_row_step_from_queue does,
    snapshotting each row from the clean buffer."""
    config = pp.create_master_array_config(metadata)
    state = pp.initialize_processing_state(config)
    state.clean_current = np.full_like(state.current_array, np.nan)
    state.clean_next = np.full_like(state.next_array, np.nan)
    row_ids = sorted(metadata["rows"])

    def bundles(r):
        return [{"skycell_id": n, "x_coord": x, "combined_image": images[n],
                 "combined_mask": np.zeros((H, W), np.uint16)} for n, x in metadata["rows"][r]]

    snapshots = {}
    for i, r in enumerate(row_ids):
        nxt = row_ids[i + 1] if i + 1 < len(row_ids) else None
        if state.current_row_id != r:
            pos, _ = pp.assemble_row_from_bundles(state.current_array, bundles(r), config)
            state.cell_locations.update(pos)
            state.current_placed = set(pos)
            state.current_row_id = r
            np.copyto(state.clean_current, state.current_array)
        if nxt is not None:
            pos, _ = pp.assemble_row_from_bundles(state.next_array, bundles(nxt), config)
            state.next_cell_locations.update(pos)
            state.next_placed = set(pos)
            state.next_row_id = nxt
            np.copyto(state.clean_next, state.next_array)
        else:
            state.next_array.fill(np.nan)
            state.clean_next.fill(np.nan)
            state.next_row_id = None
        pp.apply_cross_row_padding(state, config)

        snap = state.clean_current.copy()
        nan_mask = np.isnan(snap)
        snap[nan_mask] = 0.0
        conv = convolution_utils.apply_gaussian_convolution(snap, sigma=SIGMA, radius=RADIUS)
        conv[nan_mask] = np.nan
        snapshots.update(pp.extract_cell_results(conv, state.cell_locations))

        if junk_cross_projection:
            # Step 4 writes cross-projection patches into BOTH live buffers,
            # including the next row's (the source of problem A).
            state.current_array[:, :PAD_JUNK] = 1e6
            state.next_array[:, :PAD_JUNK] = 1e6
        if nxt is not None:
            pp.advance_sliding_window(state)
    return snapshots


PAD_JUNK = 300


def test_constants_match_producer():
    assert (pp.PAD_SIZE, pp.CELL_OVERLAP, pp.EDGE_EXCLUSION, pp.EFFECTIVE_OVERLAP) == (
        cc.PAD_SIZE, cc.CELL_OVERLAP, cc.EDGE_EXCLUSION, cc.EFFECTIVE_OVERLAP)


def test_row_path_snapshot_equals_reference_for_every_cell():
    md, imgs = _metadata(), _images()
    snaps = _run_row_path(md, imgs)
    assert set(snaps) == set(imgs)
    for name in imgs:
        ref = cc.canonical_cell_image(name, md, imgs.get, SIGMA, RADIUS)
        np.testing.assert_array_equal(np.isnan(snaps[name]), np.isnan(ref), err_msg=name)
        ok = np.isfinite(ref)
        np.testing.assert_allclose(snaps[name][ok], ref[ok], rtol=0, atol=1e-5, err_msg=name)


def test_cross_projection_junk_never_reaches_snapshot():
    md, imgs = _metadata(), _images()
    with_junk = _run_row_path(md, imgs, junk_cross_projection=True)
    without = _run_row_path(md, imgs, junk_cross_projection=False)
    for name in imgs:
        np.testing.assert_array_equal(with_junk[name], without[name], err_msg=name)


def test_last_row_top_strip_not_blanked():
    md, imgs = _metadata(), _images()
    snaps = _run_row_path(md, imgs)
    top = snaps["skycell.9999.021"][H - 10:H - 1, 5:W - 5]  # top 10 rows of a last-row cell
    assert np.isfinite(top).all()


def test_rows_with_different_first_x_are_column_aligned():
    md = _metadata()
    # cell (row 1, x=1) is vertically above (row 0, x=1): its bottom pad must come from 9999.001
    names = cc.canonical_neighbour_names(md, "skycell.9999.011")
    assert "skycell.9999.001" in names and "skycell.9999.021" in names
    x0_r0 = cc.cell_master_x0(1, cc.projection_anchor_x(md), W)
    assert x0_r0 == cc.PAD_SIZE + (1 - 0) * (W - cc.CELL_OVERLAP)


def test_neighbour_set_depends_on_mapping_list():
    md = _metadata()
    full = cc.canonical_neighbour_names(md, "skycell.9999.011")
    reduced = dict(md)
    reduced["rows"] = {r: [c for c in cells if c[0] != "skycell.9999.012"] for r, cells in md["rows"].items()}
    assert "skycell.9999.012" in full
    assert "skycell.9999.012" not in cc.canonical_neighbour_names(reduced, "skycell.9999.011")


def test_sparse_path_equals_reference():
    md, imgs = _metadata(), _images()
    bundles = {n: {"combined_image": im, "combined_mask": np.zeros((H, W), np.uint16)} for n, im in imgs.items()}
    out = pp.convolve_single_skycell("skycell.9999.011", 1, 1, "9999", md, bundles.get, SIGMA, RADIUS)
    ref = cc.canonical_cell_image("skycell.9999.011", md, imgs.get, SIGMA, RADIUS)
    np.testing.assert_array_equal(out["combined_image"], ref)
    # all-or-nothing: a missing canonical neighbour declines the sparse path
    partial = dict(bundles)
    partial.pop("skycell.9999.012")
    assert pp.convolve_single_skycell("skycell.9999.011", 1, 1, "9999", md, partial.get, SIGMA, RADIUS) is None


def test_band_weights_are_applied():
    from syndiff_pipeline.template_creation.processing.band_utils import process_skycell_bands

    rng = np.random.default_rng(3)
    bands = {b: rng.uniform(1, 2, (8, 8)).astype(np.float32) for b in "rizy"}
    d13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}
    img, _, _ = process_skycell_bands(bands, band_weights=d13)  # no headers -> no flux conversion
    expected = sum(d13[b] * bands[b] for b in "rizy")
    np.testing.assert_allclose(img, expected, rtol=1e-6)
    default, _, _ = process_skycell_bands(bands)
    assert not np.allclose(img, default)


def test_band_weights_config_validation_and_recipe():
    from syndiff_pipeline.template_creation.orchestration.stage_params import Ps1ProcessStageParams
    from syndiff_pipeline.template_creation.processing.combined_store import (
        DEFAULT_BAND_WEIGHTS, production_combined_recipe)

    assert production_combined_recipe(Ps1ProcessStageParams())["band_weights"] == DEFAULT_BAND_WEIGHTS
    d13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}
    assert production_combined_recipe(Ps1ProcessStageParams(band_weights=d13))["band_weights"] == d13
    with pytest.raises(ValueError):
        Ps1ProcessStageParams(band_weights={"r": 1.0, "i": 1.0, "z": 1.0})
    with pytest.raises(ValueError):
        Ps1ProcessStageParams(band_weights={"r": 1.0, "i": 1.0, "z": 1.0, "y": 0.0})
