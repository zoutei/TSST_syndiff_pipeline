# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Post-crop ePSF stamp renormalization."""

from __future__ import annotations

import numpy as np


def renormalize_epsf_stamp_after_crop(stamp: np.ndarray, oversampling: int) -> np.ndarray:
    """Unit native flux after border crop (stamp is on oversampled grid)."""
    s = float(np.sum(stamp))
    if s <= 0 or not np.isfinite(s):
        return stamp
    target = float(oversampling) ** 2
    return np.asarray(stamp, dtype=np.float64) / (s / target)


def apply_border_crop(stamp: np.ndarray, border_crop: int) -> np.ndarray:
    """Symmetric trim of oversampled stamp edges."""
    bc = int(border_crop)
    if bc <= 0:
        return np.asarray(stamp, dtype=np.float64)
    arr = np.asarray(stamp, dtype=np.float64)
    if bc * 2 >= arr.shape[0] or bc * 2 >= arr.shape[1]:
        return arr
    return arr[bc:-bc, bc:-bc]
