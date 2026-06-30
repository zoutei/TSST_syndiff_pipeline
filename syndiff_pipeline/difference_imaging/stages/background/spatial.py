"""Per-epoch spatial background via photutils Background2D."""

from __future__ import annotations

import logging
import multiprocessing
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from syndiff_pipeline.difference_imaging.stages.kernel_photutils import (
    photutils_background_masked,
)

log = logging.getLogger(__name__)


def _effective_n_jobs(n_jobs: int) -> int:
    if n_jobs is None or n_jobs < 1:
        return multiprocessing.cpu_count()
    return int(n_jobs)


def _spatial_one_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    box_size: int,
    filter_size: int,
    exclude_percentile: float,
    exclude_straps: bool,
) -> np.ndarray:
    phot_mask = np.asarray(mask)
    if phot_mask.ndim == 3:
        phot_mask = phot_mask[0]
    if exclude_straps:
        phot_mask = phot_mask.copy()
        phot_mask[(phot_mask.astype(np.int64) & 4) > 0] = 8
    return photutils_background_masked(
        frame,
        phot_mask,
        box_size=box_size,
        filter_size=filter_size,
        exclude_percentile=exclude_percentile,
    )


def spatial_step(
    flux_cube: np.ndarray,
    mask: np.ndarray,
    *,
    box_size: int = 16,
    filter_size: int = 3,
    exclude_percentile: float = 50.0,
    exclude_straps: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """Estimate per-frame 2D background maps for a (T, ny, nx) flux cube."""
    t, ny, nx = flux_cube.shape
    out = np.zeros((t, ny, nx), dtype=np.float64)
    nj = _effective_n_jobs(n_jobs)
    kw = dict(
        box_size=box_size,
        filter_size=filter_size,
        exclude_percentile=exclude_percentile,
        exclude_straps=exclude_straps,
    )

    if nj == 1 or t < 2:
        for i in range(t):
            out[i] = _spatial_one_frame(flux_cube[i], mask, **kw)
        return out.astype(np.float32)

    frames = Parallel(n_jobs=nj)(
        delayed(_spatial_one_frame)(flux_cube[i], mask, **kw) for i in range(t)
    )
    return np.asarray(frames, dtype=np.float32)
