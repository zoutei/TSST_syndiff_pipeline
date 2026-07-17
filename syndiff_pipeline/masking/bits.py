"""Shared-mask bit layout and consumer predicates."""

from __future__ import annotations

import numpy as np

BRIGHT_CAT = 1
SAT_CROSS = 2
STRAP = 4
EDGE = 8
PS1 = 16
FAINT_CAT = 32
TNS = 64
ASTEROID = 128

EPSF_IGNORE_BITS = BRIGHT_CAT | SAT_CROSS  # 1 | 2
STRAP_SOURCE_BITS = BRIGHT_CAT | FAINT_CAT  # 1 | 32


def epsf_reject_mask(mask: np.ndarray) -> np.ndarray:
    """True where ePSF should reject (any bit except BRIGHT_CAT / SAT_CROSS)."""
    return (np.asarray(mask).astype(np.int64) & ~EPSF_IGNORE_BITS) != 0


def full_mask_bool(mask: np.ndarray) -> np.ndarray:
    """Boolean Hotpants / photutils mask: any set bit."""
    return np.asarray(mask) != 0


def strap_source_mask(mask: np.ndarray) -> np.ndarray:
    """Catalog sources for strap QE / phot_mask (bits 1|32)."""
    return (np.asarray(mask).astype(np.int64) & STRAP_SOURCE_BITS) != 0


def strap_column_mask(mask: np.ndarray) -> np.ndarray:
    """Strap columns (bit 4)."""
    return (np.asarray(mask).astype(np.int64) & STRAP) != 0
