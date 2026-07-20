"""Mag-binned square stamps for catalog stars (bit 32 ``FAINT_CAT``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from syndiff_pipeline.difference_imaging.masking.geometry import size_limit

# Magnitude bins and square side lengths.
# Dict keys are str(mag_hi) matching the historical list form [hi, lo].
_MAG_BINS: list[tuple[float, float]] = [
    (18, 17),
    (17, 16),
    (16, 15),
    (15, 14),
    (14, 13.5),
    (13.5, 12),
    (12, 10),
    (10, 9),
    (9, 8),
    (8, 7),
]
_MAG_HI = np.array([h for h, _ in _MAG_BINS], dtype=np.float64)
_MAG_LO = np.array([lo for _, lo in _MAG_BINS], dtype=np.float64)
_SIZE_BASE = np.array([3, 4, 5, 6, 7, 8, 10, 14, 16, 18], dtype=np.int64)
_BIN_KEYS = [str(h) for h, _ in _MAG_BINS]


@njit(cache=True)
def _square_half_extents(
    mags: np.ndarray,
    mag_hi: np.ndarray,
    mag_lo: np.ndarray,
    sizes: np.ndarray,
) -> tuple:
    """
    Map each star mag → (half_lo, half_hi) matching ``fftconvolve`` of ones((sz,sz)).

    For side length ``sz``, SciPy ``mode='same'`` paints
    ``[c - (sz-1)//2, c + sz//2]`` inclusive on each axis.
    """
    n = mags.shape[0]
    half_lo = np.zeros(n, dtype=np.int64)
    half_hi = np.zeros(n, dtype=np.int64)
    active = np.zeros(n, dtype=np.uint8)
    n_bins = mag_hi.shape[0]
    for i in range(n):
        m = mags[i]
        for b in range(n_bins):
            if m > mag_lo[b] and m <= mag_hi[b]:
                sz = sizes[b]
                if sz > 0:
                    half_lo[i] = (sz - 1) // 2
                    half_hi[i] = sz // 2
                    active[i] = 1
                break
    return half_lo, half_hi, active


@njit(parallel=True, cache=True)
def paint_squares(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    half_lo: np.ndarray,
    half_hi: np.ndarray,
    active: np.ndarray,
) -> None:
    """Paint axis-aligned squares onto uint8 mask in-place (FFT-equivalent extent)."""
    ny, nx = mask.shape
    n = xs.shape[0]
    for i in prange(n):
        if active[i] == 0:
            continue
        xi = xs[i]
        yi = ys[i]
        lo = half_lo[i]
        hi = half_hi[i]
        y0 = yi - lo
        if y0 < 0:
            y0 = 0
        y1 = yi + hi + 1
        if y1 > ny:
            y1 = ny
        x0 = xi - lo
        if x0 < 0:
            x0 = 0
        x1 = xi + hi + 1
        if x1 > nx:
            x1 = nx
        for y in range(y0, y1):
            for x in range(x0, x1):
                mask[y, x] = 1


def faint_star_squares(
    table: pd.DataFrame, Image: np.ndarray, scale: float = 1.0
) -> dict:
    """
    Build a magnitude-keyed square-stamp mask dict from a star catalog.

    Each magnitude bin gets a square stamp of increasing size, painted with
    numba (FFT-equivalent footprint to the historical ``fftconvolve`` path).

    Expects table columns: x, y, mag (crop-local pixels).
    Returns dict with per-bin keys and ``'all'`` (union).
    """
    image = np.zeros_like(Image)
    x = np.round(table["x"].to_numpy(float), 0).astype(np.int64)
    y = np.round(table["y"].to_numpy(float), 0).astype(np.int64)
    m = table["mag"].to_numpy(float).astype(np.float64)
    ind = size_limit(x, y, image)
    x, y, m = x[ind], y[ind], m[ind]

    sizes = np.maximum(
        (_SIZE_BASE.astype(np.float64) * float(scale)).astype(np.int64), 0
    )
    half_lo, half_hi, active = _square_half_extents(m, _MAG_HI, _MAG_LO, sizes)

    masks: dict = {}
    for b, key in enumerate(_BIN_KEYS):
        bin_mask = np.zeros(image.shape, dtype=np.uint8)
        if len(x):
            sel = (active > 0) & (m > _MAG_LO[b]) & (m <= _MAG_HI[b])
            if np.any(sel):
                n_sel = int(sel.sum())
                paint_squares(
                    bin_mask,
                    x[sel],
                    y[sel],
                    half_lo[sel],
                    half_hi[sel],
                    np.ones(n_sel, dtype=np.uint8),
                )
        masks[key] = bin_mask.astype(np.float64)

    all_mask = np.zeros(image.shape, dtype=np.uint8)
    if len(x):
        paint_squares(all_mask, x, y, half_lo, half_hi, active)
    masks["all"] = all_mask.astype(np.float64)
    return masks
