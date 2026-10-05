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


def _fake_projection_catalog(data_root, projection="9999"):
    """Write only what the fingerprint reads: the projection catalogue's meta file."""
    import json

    from syndiff_pipeline.template_creation.processing import gaia_projection_catalog as gpc

    path = gpc.projection_catalog_path(data_root, projection)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_name(path.name + ".meta.json")
    meta.write_text(json.dumps({"content_sha256": "ab" * 32, "scheme": gpc.GAIA_PROJECTION_SCHEME}))


def _mapping_df():
    import pandas as pd

    rows = [{"NAME": n, "projection": 9999, "y": r, "x": x, "NAXIS1": W, "NAXIS2": H}
            for r, cells in ROWS.items() for n, x in cells]
    return pd.DataFrame(rows)


def test_end_to_end_store_publish_resolve_and_spot_check(tmp_path):
    """Combined publish (coordinator helper) -> row-path canonical publish (real
    function) -> reader resolves the v2 fingerprint from its own mapping list ->
    spot check recomputes identical pixels. A reader with a different mapping
    list does not resolve it, and different pixels under the same fingerprint raise."""
    from syndiff_pipeline.template_creation.orchestration.stage_params import Ps1ProcessStageParams
    from syndiff_pipeline.template_creation.processing import convolved_store
    from syndiff_pipeline.template_creation.processing.combined_store import (
        expected_combined_fingerprint,
        production_combined_recipe,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import _discover_shared_convolved_fp

    data_root = tmp_path / "data"
    _fake_projection_catalog(data_root)
    md, imgs = _metadata(), _images()
    df = _mapping_df()
    crecipe = production_combined_recipe(Ps1ProcessStageParams())
    vrecipe = convolved_store.convolved_recipe(psf_sigma=SIGMA, radius=RADIUS)
    masks = {n: np.zeros((H, W), np.uint16) for n in imgs}
    for n, im in imgs.items():
        info = pp.publish_combined_result(str(data_root), crecipe, {
            "skycell_id": n, "combined_image": im, "combined_mask": masks[n],
            "headers_data": {}, "removed_stars": []})
        assert info is not None

    config = pp.create_master_array_config(md)
    state = pp.initialize_processing_state(config)
    state.clean_current = np.full_like(state.current_array, np.nan)
    state.clean_next = np.full_like(state.next_array, np.nan)
    row_ids = sorted(md["rows"])

    def bundles(r):
        return [{"skycell_id": n, "x_coord": x, "combined_image": imgs[n], "combined_mask": masks[n]}
                for n, x in md["rows"][r]]

    for i, r in enumerate(row_ids):
        nxt = row_ids[i + 1] if i + 1 < len(row_ids) else None
        if state.current_row_id != r:
            pos, m = pp.assemble_row_from_bundles(state.current_array, bundles(r), config)
            state.cell_locations.update(pos)
            state.current_masks.update(m)
            state.cell_metadata.update({n: {"headers_data": {}, "removed_stars": []} for n in pos})
            state.current_placed = set(pos)
            state.current_row_id = r
            np.copyto(state.clean_current, state.current_array)
        if nxt is not None:
            pos, m = pp.assemble_row_from_bundles(state.next_array, bundles(nxt), config)
            state.next_cell_locations.update(pos)
            state.next_masks.update(m)
            state.next_cell_metadata.update({n: {"headers_data": {}, "removed_stars": []} for n in pos})
            state.next_placed = set(pos)
            state.next_row_id = nxt
            np.copyto(state.clean_next, state.next_array)
        else:
            state.next_array.fill(np.nan)
            state.clean_next.fill(np.nan)
            state.next_row_id = None
        pp.apply_cross_row_padding(state, config)
        pp._publish_canonical_convolved_snapshot(state, "9999", SIGMA, str(data_root), crecipe, vrecipe, metadata=md)
        state.current_array[:, :PAD_JUNK] = 1e6  # cross-projection patches (step 4)
        state.next_array[:, :PAD_JUNK] = 1e6
        if nxt is not None:
            pp.advance_sliding_window(state)

    fps = {}
    for n in imgs:
        projection, cell = n.rsplit(".", 1)
        fps[n] = cc.resolve_canonical_convolved_fp(data_root, n, md, crecipe, vrecipe)
        assert fps[n] is not None, n
        # the downsample reader without the mapping list refuses (neighbour set unknown)
        assert _discover_shared_convolved_fp(data_root, projection, cell, psf_sigma=SIGMA,
                                             combined_recipe=crecipe) is None

    checks = cc.spot_check_cells(data_root, sorted(imgs), df, crecipe, vrecipe)
    assert [c["status"] for c in checks] == ["ok"] * len(imgs), checks
    assert max(c["max_rel"] for c in checks) <= 1e-6

    # a reader whose mapping list lacks 012 must not resolve 011's stored cell
    reduced = df[df.NAME != "skycell.9999.012"].reset_index(drop=True)
    md_reduced = cc.metadata_for_cell(reduced, "skycell.9999.011")
    assert cc.resolve_canonical_convolved_fp(data_root, "skycell.9999.011", md_reduced, crecipe, vrecipe) is None

    # different pixels under an existing fingerprint raise
    n = "skycell.9999.011"
    projection, cell = n.rsplit(".", 1)
    stored = convolved_store.try_load_convolved_cell(data_root, projection, cell, fps[n])
    with pytest.raises(convolved_store.ConvolvedFingerprintConflict):
        convolved_store.publish_convolved_cell(
            data_root, projection, cell, convolved_image=stored["convolved_image"] * 1.01,
            convolved_mask=stored["convolved_mask"], headers_data={}, removed_stars=[], recipe=vrecipe,
            combined_fingerprint=expected_combined_fingerprint(data_root, projection, cell, crecipe),
            extra_input_fingerprints=cc.neighbour_input_fingerprints(data_root, n, md, crecipe))


def test_combined_publish_refuses_without_projection_catalog(tmp_path):
    from syndiff_pipeline.template_creation.orchestration.stage_params import Ps1ProcessStageParams
    from syndiff_pipeline.template_creation.processing.combined_store import production_combined_recipe

    crecipe = production_combined_recipe(Ps1ProcessStageParams())
    info = pp.publish_combined_result(str(tmp_path), crecipe, {
        "skycell_id": "skycell.9999.011", "combined_image": np.zeros((4, 4), np.float32),
        "combined_mask": np.zeros((4, 4), np.uint16), "headers_data": {}, "removed_stars": []})
    assert info is None


# ---------------------------------------------------------------------------
# v3: one writer for vertical overlaps (dev_runs/spike_diag_20261001)
# ---------------------------------------------------------------------------


def _global_offsets(md):
    """(row0, col0) of each cell in one projection-wide frame: rows and columns
    advance by (size - CELL_OVERLAP)."""
    return {n: (int(r) * (H - cc.CELL_OVERLAP), int(x) * (W - cc.CELL_OVERLAP))
            for r, cells in md["rows"].items() for n, x in cells}


def _overlap_pairs(md):
    off = _global_offsets(md)
    names = sorted(off)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (ra, ca), (rb, cb) = off[a], off[b]
            r0, r1 = max(ra, rb), min(ra, rb) + H
            c0, c1 = max(ca, cb), min(ca, cb) + W
            if r1 > r0 and c1 > c0:
                yield (a, (slice(r0 - ra, r1 - ra), slice(c0 - ca, c1 - ca)),
                       b, (slice(r0 - rb, r1 - rb), slice(c0 - cb, c1 - cb)), ra != rb)


def test_overlapping_cells_agree_on_shared_sky():
    """Every same-projection canonical cell is a crop of one projection image:
    any two cells hold the same convolved pixels where they overlap, in rows
    (horizontal), between rows (vertical, the v3 change) and diagonally. The
    synthetic cells' own pixels differ everywhere (independent noise), as real
    cells do where star removal or the PS1 stack differs."""
    md, imgs = _metadata(), _images()
    ref = {n: cc.canonical_cell_image(n, md, imgs.get, SIGMA, RADIUS) for n in imgs}
    n_vertical = 0
    for a, sa, b, sb, vertical in _overlap_pairs(md):
        assert not np.allclose(imgs[a][sa], imgs[b][sb], equal_nan=True)  # inputs disagree
        np.testing.assert_array_equal(np.isnan(ref[a][sa]), np.isnan(ref[b][sb]), err_msg=f"{a}/{b}")
        np.testing.assert_allclose(ref[a][sa], ref[b][sb], rtol=0, atol=1e-5, equal_nan=True, err_msg=f"{a}/{b}")
        n_vertical += vertical
    assert n_vertical >= 4


def test_vertical_strip_comes_from_lower_row():
    """Row R's shared strip holds row R-1's pixels (checked before blur, radius 0)."""
    md, imgs = _metadata(), _images()
    upper = cc.canonical_cell_image("skycell.9999.011", md, imgs.get, 1e-6, 0)  # row 1, x=1
    lower = imgs["skycell.9999.001"]                                         # row 0, x=1
    strip = slice(cc.EDGE_EXCLUSION, cc.EFFECTIVE_OVERLAP)
    lower_rows = slice(H - cc.CELL_OVERLAP + cc.EDGE_EXCLUSION, H - cc.CELL_OVERLAP + cc.EFFECTIVE_OVERLAP)
    cols = slice(5, cc.EFFECTIVE_OVERLAP)  # inside 001, left of 002's columns
    np.testing.assert_allclose(upper[strip, cols], lower[lower_rows, cols], rtol=0, atol=1e-5, equal_nan=True)


def test_missing_lower_neighbour_keeps_own_strip():
    """Row 0 has no x=0 cell: 9999.010 (row 1, x=0) keeps its own pixels in the
    strip where nothing lies below (not NaN, not another cell's pixels)."""
    md, imgs = _metadata(), _images()
    out = cc.canonical_cell_image("skycell.9999.010", md, imgs.get, 1e-6, 0)
    strip = slice(cc.EDGE_EXCLUSION, cc.EFFECTIVE_OVERLAP)
    cols = slice(5, W - cc.CELL_OVERLAP - 5)  # x=0 columns left of the x=1 cell
    np.testing.assert_allclose(out[strip, cols], imgs["skycell.9999.010"][strip, cols], rtol=0, atol=1e-5)


def test_cross_row_copies_need_column_masks():
    cur = np.zeros((H + 2 * cc.PAD_SIZE, 10), np.float32)
    with pytest.raises(ValueError):
        cc.apply_cross_row(cur, cur.copy(), None, H)
    with pytest.raises(ValueError):
        cc.apply_cross_row(cur, None, cur.copy(), H)


# Real-data regression on the paper dataset's F2 cells (s0020 c3 k3), skipped
# when that data is absent. The combined cells are the published ones (fixed
# inputs); the canonical images are recomputed with this code.
_PAPER = "/astro/armin/koji/syndiff/dev_runs/paper_dataset_20261001"
_D13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}


_PAPER_COMBINED_RECIPE_ID = "e17a198a4942aa2d"  # the dataset's published combined cells (D13, footprint_v1)


def _paper_combined(root, name):
    """The dataset's published combined cell via its current pointer, checked to be recipe e17a198a (fixed
    inputs: production now mints a different combined recipe, so it cannot resolve these cells)."""
    import json
    import os

    from syndiff_pipeline.template_creation.processing.combined_store import try_load_combined_cell

    projection, cell = name.rsplit(".", 1)
    ptr = f"{root}/ps1_skycells_zarr/ps1_combined.zarr/{projection}/{cell}/current.json"
    if not os.path.exists(ptr):
        return None
    with open(ptr) as fh:
        cur = json.load(fh)
    if cur.get("recipe_id") != _PAPER_COMBINED_RECIPE_ID:
        return None
    return try_load_combined_cell(root, projection, cell, cur["fingerprint"])


def _paper_case(names):
    import os

    import pandas as pd

    root = f"{_PAPER}/data_root"
    lst = f"{root}/s0020/c3/k3/mapping/oversampling_1/tess_s0020_3_3_master_skycells_list.csv"
    if not os.path.exists(lst):
        pytest.skip("paper dataset not available")
    df = pd.read_csv(lst)
    out = {}
    for n in names:
        md = cc.metadata_for_cell(df, n)
        needed = [n, *cc.canonical_neighbour_names(md, n)]
        cache = {k: c for k in needed if (c := _paper_combined(root, k)) is not None}
        if any(k not in cache for k in needed):
            pytest.fail(f"combined inputs of {n} missing: {[k for k in needed if k not in cache]}")
        fetch = lambda k, c=cache: None if k not in c else np.asarray(c[k]["combined_image"], np.float32)
        out[n] = (cc.canonical_cell_image(n, md, fetch, 40.0, 470), cache[n]["headers_data"])
    return out


def _box_at(img, headers, ra, dec, half=40):
    from astropy.io import fits
    from astropy.wcs import WCS

    h = headers["r"] if "r" in headers else next(iter(headers.values()))
    x, y = WCS(fits.Header.fromstring(h)).all_world2pix([[ra, dec]], 0)[0]
    xi, yi = int(round(x)), int(round(y))
    return img[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float64)


def test_paper_f2_vertical_spike_star_is_whole_in_both_cells():
    """F2 spike (1253,711): 2611.013 (row 1) kept a T 13.8 star that 2611.023
    (row 2) zeroed as a bright star's catalog_neighbor. The v2 canonical cells
    held 7649 vs 92 there (81x81 box); v3 cells both hold the lower row's star."""
    cells = _paper_case(["skycell.2611.013", "skycell.2611.023"])
    ra, dec = 225.46428, 76.76520
    a = _box_at(*cells["skycell.2611.013"], ra, dec)
    b = _box_at(*cells["skycell.2611.023"], ra, dec)
    assert a.sum() > 5000
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-6 * np.abs(a).max())


@pytest.mark.xfail(strict=True, reason="known limitation: cross-projection overlaps are different PS1 stacks "
                                       "(2611.091 removed the T=13.31 star via its own saturation flags)")
def test_paper_f2_cross_projection_pair_agrees():
    cells = _paper_case(["skycell.2611.091", "skycell.2612.098"])
    ra, dec = 231.662963, 79.751358
    a = _box_at(*cells["skycell.2612.098"], ra, dec)
    b = _box_at(*cells["skycell.2611.091"], ra, dec)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-3 * np.abs(a).max())
