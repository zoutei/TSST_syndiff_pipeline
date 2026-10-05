"""perband.blur_cell_row_path IS the canonical cell (schema v3) for one plane."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.template_creation.processing import canonical_cell as cc
from syndiff_pipeline.template_creation.processing import perband

W = H = 1000
SIGMA, RADIUS = 5.0, 20

# Ragged rows (different first x, missing neighbours) exercise the vertical one-writer rule.
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
            img[0, :] = np.nan
            img[:, -1] = np.nan
            out[name] = img
    return out


def test_blur_cell_row_path_equals_canonical_cell():
    md, imgs = _metadata(), _images()
    for name in imgs:
        got = perband.blur_cell_row_path(name, md, imgs.get, SIGMA, RADIUS)
        ref = cc.canonical_cell_image(name, md, imgs.get, SIGMA, RADIUS)
        np.testing.assert_array_equal(got, ref, err_msg=name)


def test_band_sum_equals_canonical_of_summed_bands():
    md, imgs = _metadata(), _images()
    weights = (0.2, 0.3, 0.25, 0.25)
    rng = np.random.default_rng(11)
    bands = {}
    for b, w in zip(perband.BANDS, weights):
        bands[b] = {
            n: (w * img + 0.01 * rng.normal(size=img.shape)).astype(np.float32)
            for n, img in imgs.items()
        }
    summed = {n: sum(bands[b][n] for b in perband.BANDS).astype(np.float32) for n in imgs}

    def fetch_cells(name):
        return {b: bands[b][name] for b in perband.BANDS}

    for name in imgs:
        blurred = perband.convolve_band_cells(name, md, fetch_cells, SIGMA, RADIUS)
        tot = sum(blurred[b].astype(np.float64) for b in perband.BANDS)
        ref = cc.canonical_cell_image(name, md, summed.get, SIGMA, RADIUS)
        np.testing.assert_array_equal(np.isnan(tot), np.isnan(ref), err_msg=name)
        ok = np.isfinite(ref)
        np.testing.assert_allclose(tot[ok], ref[ok], rtol=0, atol=1e-4, err_msg=name)


def test_cell_not_in_metadata_returns_none():
    md, imgs = _metadata(), _images()
    assert perband.blur_cell_row_path("skycell.9999.999", md, imgs.get, SIGMA, RADIUS) is None
