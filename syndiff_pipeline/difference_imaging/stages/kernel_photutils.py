"""Masked photutils Background2D for kernel-based differencing."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def photutils_background_masked(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    box_size: int = 4,
    filter_size: int = 3,
    exclude_percentile: float = 50.0,
) -> np.ndarray:
    """
    Estimate a 2D background with photutils Background2D on unmasked pixels.

    ``mask`` uses the shared-mask convention: any value > 0 excludes the pixel.
    """
    from photutils.background import Background2D, MedianBackground

    frame = np.asarray(image, dtype=np.float64)
    mask2d = np.asarray(mask)
    if mask2d.ndim == 3:
        mask2d = mask2d[0]
    phot_mask = mask2d.astype(bool) if mask2d.dtype == bool else (mask2d > 0)

    ny, nx = frame.shape
    eff_box = max(4, min(int(box_size), min(ny, nx) // 2))

    try:
        bkg2d = Background2D(
            frame,
            box_size=eff_box,
            filter_size=int(filter_size),
            mask=phot_mask,
            bkg_estimator=MedianBackground(),
            exclude_percentile=float(exclude_percentile),
        )
        return np.asarray(bkg2d.background, dtype=np.float64)
    except Exception as exc:
        log.warning("Background2D failed (%s); using nanmedian fallback", exc)
        good = ~phot_mask & np.isfinite(frame)
        fill = float(np.nanmedian(frame[good])) if good.any() else 0.0
        return np.full(frame.shape, fill, dtype=np.float64)
