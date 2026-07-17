"""Numba faint_star_squares footprints match historical fftconvolve stamps."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from syndiff_pipeline.masking.faint_star_squares import (
    _BIN_KEYS,
    _MAG_HI,
    _MAG_LO,
    _SIZE_BASE,
    faint_star_squares,
)
from syndiff_pipeline.masking.geometry import size_limit


def _faint_star_squares_fft(
    table: pd.DataFrame, image: np.ndarray, scale: float = 1.0
) -> dict:
    """Historical FFT reference (equivalence tests only)."""
    canvas = np.zeros_like(image)
    x = (np.round(table["x"].values, 0)).astype(int)
    y = (np.round(table["y"].values, 0)).astype(int)
    m = table["mag"].values
    ind = size_limit(x, y, canvas)
    x, y, m = x[ind], y[ind], m[ind]

    masks = {}
    mags = [[hi, lo] for hi, lo in zip(_MAG_HI.tolist(), _MAG_LO.tolist())]
    sizes = (_SIZE_BASE.astype(float) * scale).astype(int)
    for i, mag_range in enumerate(mags):
        mag_ind = (m > mag_range[1]) & (m <= mag_range[0])
        magim = np.zeros_like(canvas)
        magim[y[mag_ind], x[mag_ind]] = 1.0
        sz = sizes[i]
        if sz > 0:
            conv = fftconvolve(magim, np.ones((sz, sz)), mode="same")
            masks[_BIN_KEYS[i]] = (conv > 0.1) * 1.0
        else:
            masks[_BIN_KEYS[i]] = np.zeros_like(canvas)
    masks["all"] = np.zeros_like(canvas, dtype=float)
    for key in masks:
        masks["all"] += masks[key]
    masks["all"] = (masks["all"] > 0.1) * 1.0
    return masks


def test_faint_star_squares_matches_fft_footprint():
    rng = np.random.default_rng(0)
    shape = (128, 128)
    image = np.zeros(shape)
    n = 200
    table = pd.DataFrame(
        {
            "x": rng.integers(5, shape[1] - 5, size=n).astype(float),
            "y": rng.integers(5, shape[0] - 5, size=n).astype(float),
            "mag": rng.uniform(7.1, 18.0, size=n),
        }
    )
    numba_m = faint_star_squares(table, image, scale=1.0)
    fft_m = _faint_star_squares_fft(table, image, scale=1.0)
    assert set(numba_m) == set(fft_m)
    for key in numba_m:
        assert np.array_equal(
            numba_m[key].astype(bool), fft_m[key].astype(bool)
        ), f"mismatch on key={key}"


def test_faint_star_squares_even_odd_single_star():
    image = np.zeros((40, 40))
    for mag, sz, half_lo, half_hi in [
        (17.5, 3, 1, 1),
        (16.5, 4, 1, 2),
    ]:
        table = pd.DataFrame({"x": [20.0], "y": [20.0], "mag": [mag]})
        m = faint_star_squares(table, image)["all"].astype(bool)
        ys, xs = np.where(m)
        assert ys.min() == 20 - half_lo
        assert ys.max() == 20 + half_hi
        assert xs.min() == 20 - half_lo
        assert xs.max() == 20 + half_hi
        assert m.sum() == sz * sz


def test_gaia_auto_mask_alias():
    from syndiff_pipeline.masking.tessreduce_squares import gaia_auto_mask

    assert gaia_auto_mask is faint_star_squares
