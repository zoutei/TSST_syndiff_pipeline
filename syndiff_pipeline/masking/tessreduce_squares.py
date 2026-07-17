"""TESSreduce square catalog masks (gaia_auto_mask) and Big_sat crosses."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from syndiff_pipeline.masking.geometry import size_limit

log = logging.getLogger(__name__)


def gaia_auto_mask(table: pd.DataFrame, Image: np.ndarray, scale: float = 1.0) -> dict:
    """
    Build a magnitude-keyed mask dict from a Gaia catalog.
    Each magnitude bin gets a square kernel of increasing size.
    Returns dict with key 'all' containing the union mask.

    Expects table columns: x, y, mag (all in crop-local pixels).
    """
    image = np.zeros_like(Image)
    x = (np.round(table["x"].values, 0)).astype(int)
    y = (np.round(table["y"].values, 0)).astype(int)
    m = table["mag"].values
    ind = size_limit(x, y, image)
    x, y, m = x[ind], y[ind], m[ind]

    masks = {}
    mags = [
        [18, 17],
        [17, 16],
        [16, 15],
        [15, 14],
        [14, 13.5],
        [13.5, 12],
        [12, 10],
        [10, 9],
        [9, 8],
        [8, 7],
    ]
    sizes = (np.array([3, 4, 5, 6, 7, 8, 10, 14, 16, 18]) * scale).astype(int)

    for i, mag_range in enumerate(mags):
        mag_ind = (m > mag_range[1]) & (m <= mag_range[0])
        magim = np.zeros_like(image)
        magim[y[mag_ind], x[mag_ind]] = 1.0
        sz = sizes[i]
        if sz > 0:
            k = np.ones((sz, sz))
            conv = fftconvolve(magim, k, mode="same")
            masks[str(mag_range[0])] = (conv > 0.1) * 1.0

    masks["all"] = np.zeros_like(image, dtype=float)
    for key in masks:
        masks["all"] += masks[key]
    masks["all"] = (masks["all"] > 0.1) * 1.0
    return masks


def Big_sat(table: pd.DataFrame, Image: np.ndarray, scale: float = 1.0) -> list:
    """
    Build cross + circular body masks for stars brighter than mag 7 (TESSreduce).

    Expects table columns: x, y, mag (crop-local pixels).
    Returns list of 2D mask arrays.
    """
    image = np.zeros_like(Image)
    sat = table[table["mag"].values < 7].copy()
    x = (np.round(sat["x"].values, 0)).astype(int)
    y = (np.round(sat["y"].values, 0)).astype(int)
    m = sat["mag"].values
    ind = size_limit(x, y, image)
    x, y, m = x[ind], y[ind], m[ind]

    satmasks = []
    for i in range(len(x)):
        mag = m[i]
        mask = np.zeros_like(image, dtype=float)

        body = int(13 * scale)
        length = int(20 * scale)
        width = int(3 * scale)

        if mag <= 5 and mag > 4:
            body = int(15 * scale)
            length = int(60 * scale)
            width = int(5 * scale)
        elif mag <= 4:
            body = int(22 * scale)
            length = int(115 * scale)
            width = int(7 * scale)

        kernel = np.zeros((body * 2 + 1, body * 2 + 1))
        yy, xx = np.where(kernel == 0)
        dist = np.sqrt((yy - body) ** 2 + (xx - body) ** 2)
        kernel[yy[dist <= body + 1], xx[dist <= body + 1]] = 1
        stamp = np.zeros_like(image)
        stamp[y[i], x[i]] = 1
        conv = fftconvolve(stamp, kernel, mode="same")
        mask = (conv > 0.1) * 1.0

        for r0, r1, c0, c1 in [
            (max(0, y[i] - length), y[i] + length, max(0, x[i] - width), x[i] + width),
            (max(0, y[i] - width), y[i] + width, max(0, x[i] - length), x[i] + length),
        ]:
            mask[r0:r1, c0:c1] = 1

        satmasks.append(mask)

    return satmasks


def Strap_mask(
    Image: np.ndarray, col_offset: int, straps_csv: str, size: int = 4
) -> np.ndarray:
    """Build a strap mask for TESS CCDs (TESSreduce convention)."""
    strap_mask = np.zeros_like(Image)

    if not straps_csv or not os.path.isfile(straps_csv):
        from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
            tess_straps_csv,
        )

        straps_csv = str(tess_straps_csv())

    if not os.path.exists(straps_csv):
        log.warning("tess_straps.csv not found at %s. Strap masking disabled.", straps_csv)
        return strap_mask

    straps_df = pd.read_csv(straps_csv)
    straps = straps_df["Column"].values - col_offset + 44
    strap_in_crop = straps[(straps > 0) & (straps < Image.shape[1])]
    strap_mask[:, strap_in_crop.astype(int)] = 1

    k_size = max(1, int(size))
    if k_size % 2 == 0:
        k_size += 1
    big_strap = fftconvolve(strap_mask, np.ones((k_size, k_size)), mode="same") > 0.5
    return big_strap.astype(int)
