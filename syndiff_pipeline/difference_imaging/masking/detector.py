"""Detector strap, edge, and PS1 coverage mask helpers."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def detector_edge_mask(
    shape: tuple[int, int],
    crop_bounds: dict,
    *,
    nx: int,
    ny: int,
    x_left_dead: int = 44,
    x_right_dead: int = 44,
    y_edge_strip: int = 30,
) -> np.ndarray:
    """
    Mask TESS detector non-science regions intersecting the crop (bit 8).

    Usable FFI area is ``x in [x_left_dead, nx - x_right_dead)``,
    ``y in [0, ny - y_edge_strip)``.
    """
    ny_crop, nx_crop = shape
    x_min = int(crop_bounds["x_min"])
    y_min = int(crop_bounds["y_min"])
    x_usable_lo = int(x_left_dead)
    x_usable_hi = nx - int(x_right_dead)
    y_usable_hi = ny - int(y_edge_strip)

    edge = np.zeros((ny_crop, nx_crop), dtype=bool)
    cols = x_min + np.arange(nx_crop)
    rows = y_min + np.arange(ny_crop)
    bad_col = (cols < x_usable_lo) | (cols >= x_usable_hi)
    bad_row = rows >= y_usable_hi
    edge[:, bad_col] = True
    edge[bad_row, :] = True
    return edge


def strap_mask(
    image: np.ndarray,
    col_offset: int,
    straps_csv: str | None,
    size: int = 6,
) -> np.ndarray:
    """TESS strap columns (bit 4), empirical convention (half-width strip)."""
    from syndiff_pipeline.difference_imaging.masking.tessreduce_squares import Strap_mask

    # Prefer TESSreduce Strap_mask for identical dilation behavior
    return Strap_mask(image, col_offset, straps_csv or "", size=size).astype(int)


def ps1_coverage_mask(count_crop: np.ndarray, *, min_hit_count: int = 5000) -> np.ndarray:
    """True where PS1 hit count is below *min_hit_count*."""
    return count_crop < int(min_hit_count)


def resolve_straps_csv(straps_csv: str | None) -> str:
    if straps_csv and os.path.isfile(straps_csv):
        return straps_csv
    from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
        tess_straps_csv,
    )

    return str(tess_straps_csv())
