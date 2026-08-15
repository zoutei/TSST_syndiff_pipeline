"""Regression test for the corner-location patch-sizing bug in
``padding_correction._standalone_padding_wcs``.

Found 2026-08-10 on real CVZ data: a naive ``"top" in location`` /
``"right" in location`` pair of substring checks both fire for a corner
location like ``"top_right"`` (it contains both words), and each branch
independently overwrote width/height to the *full cell dimension* --
turning an intended ~500px square corner patch into a reprojection target
the size of the whole cell. That pasted a large fraction of the diagonal
neighbor skycell's real image on top of the recipient cell (a real
double-count of star flux at the seam, not just a wider blend), producing
duplicate/ghost stars and oversubtraction (dark) residuals in ``hp_d``
wherever a corner-adjacent cross-projection correction was applied.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing.padding_correction import (
    PAD_SIZE,
    EDGE_EXCLUSION,
    _standalone_padding_wcs,
)

CELL_WIDTH = 6481
CELL_HEIGHT = 6429
_PATCH = PAD_SIZE + EDGE_EXCLUSION


@pytest.fixture
def recipient_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [180.0, -30.0]
    wcs.wcs.crpix = [CELL_WIDTH / 2, CELL_HEIGHT / 2]
    wcs.wcs.cd = [[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.mark.parametrize(
    "location",
    ["top_left", "top_right", "bottom_left", "bottom_right"],
)
def test_corner_patch_is_small_square_not_full_cell(recipient_wcs, location):
    _, shape, _ = _standalone_padding_wcs(recipient_wcs, CELL_WIDTH, CELL_HEIGHT, location)
    assert shape == (_PATCH, _PATCH)
    assert shape != (CELL_HEIGHT, CELL_WIDTH)


@pytest.mark.parametrize("location", ["top", "bottom"])
def test_top_bottom_edge_patch_spans_full_width_thin_height(recipient_wcs, location):
    _, shape, _ = _standalone_padding_wcs(recipient_wcs, CELL_WIDTH, CELL_HEIGHT, location)
    assert shape == (_PATCH, CELL_WIDTH)


@pytest.mark.parametrize("location", ["left", "right"])
def test_left_right_edge_patch_spans_full_height_thin_width(recipient_wcs, location):
    _, shape, _ = _standalone_padding_wcs(recipient_wcs, CELL_WIDTH, CELL_HEIGHT, location)
    assert shape == (CELL_HEIGHT, _PATCH)


def test_all_eight_locations_never_exceed_patch_or_full_cell_dims(recipient_wcs):
    """Every valid location's target area must be a strict subset of a
    small patch in each corner-relevant axis -- never the full cell in
    both dimensions simultaneously (that was the bug's signature)."""
    for location in [
        "top", "bottom", "left", "right",
        "top_left", "top_right", "bottom_left", "bottom_right",
    ]:
        _, (h, w), _ = _standalone_padding_wcs(recipient_wcs, CELL_WIDTH, CELL_HEIGHT, location)
        assert not (h == CELL_HEIGHT and w == CELL_WIDTH), (
            f"{location} produced a full-cell-sized patch {(h, w)} -- "
            "this is the double-count bug"
        )
