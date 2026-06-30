"""Temporal smoothing of background stacks (SavGol or AdaptiveBackground)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from syndiff_pipeline.difference_imaging.stages.adaptive_background import (
    AdaptiveBackground,
    savgol_smooth_3d,
    savgol_smooth_3d_parallel,
)

log = logging.getLogger(__name__)


def _sanitize_btjd(btjd: np.ndarray) -> np.ndarray:
    t = np.asarray(btjd, dtype=float)
    if np.isnan(t).all():
        return np.arange(len(t), dtype=float)
    if not np.isnan(t).any():
        return t
    x = np.arange(len(t))
    m = np.isfinite(t)
    return np.interp(x, x[m], t[m])


def temporal_step(
    bkg_stack: np.ndarray,
    time_btjd: np.ndarray,
    sector: int,
    camera: int,
    *,
    method: str = "savgol",
    savgol_window: Optional[int] = None,
    savgol_polyorder: int = 2,
    gap_thresh_days: float = 0.5,
    tile_size: int = 256,
    w_min: int = 3,
    w_max: int = 51,
    block_size: int = 5,
    vector_path: Optional[str] = None,
    n_jobs: int = 1,
    sigma_clip: float = 5.0,
) -> np.ndarray:
    """Smooth a (T, ny, nx) background cube along time."""
    time_btjd = _sanitize_btjd(np.asarray(time_btjd, dtype=float))
    if time_btjd.shape[0] != bkg_stack.shape[0]:
        raise ValueError(
            f"time_btjd length {time_btjd.shape[0]} != stack T={bkg_stack.shape[0]}"
        )

    m = str(method).lower().strip()
    if m not in ("savgol", "adaptive"):
        raise ValueError(f"temporal method must be 'savgol' or 'adaptive', got {method!r}")

    if m == "adaptive":
        time_mjd = time_btjd + 57000.0
        smoother = AdaptiveBackground(
            bkg_stack,
            time_mjd,
            sector=int(sector),
            camera=int(camera),
            data_path=vector_path,
            n_jobs=n_jobs,
            block_size=block_size,
        )
        smoother.smooth(
            method="adaptive",
            w_min=w_min,
            w_max=w_max,
            n_jobs=n_jobs,
        )
        return np.asarray(smoother.smoothed, dtype=np.float32)

    time_mjd = time_btjd + 57000.0
    if tile_size <= 0 or n_jobs <= 1:
        smoothed = savgol_smooth_3d(
            bkg_stack,
            time=time_mjd,
            gap_thresh=gap_thresh_days,
            window_length=savgol_window,
            polyorder=savgol_polyorder,
            sigma_clip=sigma_clip,
        )
        return np.asarray(smoothed, dtype=np.float32)

    smoothed = savgol_smooth_3d_parallel(
        bkg_stack,
        time=time_mjd,
        gap_thresh=gap_thresh_days,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
        sigma_clip=sigma_clip,
        tile_size=tile_size,
        n_jobs=n_jobs,
    )
    return np.asarray(smoothed, dtype=np.float32)
