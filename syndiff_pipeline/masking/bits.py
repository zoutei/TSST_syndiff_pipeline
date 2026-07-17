"""Static-mask bit layout and consumer predicates."""

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

# ePSF may use catalog stars (bright circles, sat crosses, faint squares).
EPSF_IGNORE_BITS = BRIGHT_CAT | SAT_CROSS | FAINT_CAT  # 1 | 2 | 32
# Hotpants ignores Gaia faint squares (>~13); still masks bright/crosses/straps/etc.
HOTPANTS_IGNORE_BITS = FAINT_CAT  # 32
STRAP_SOURCE_BITS = BRIGHT_CAT | FAINT_CAT  # 1 | 32


def epsf_reject_mask(mask: np.ndarray) -> np.ndarray:
    """
    True where ePSF should reject a pixel for star selection.

    Ignores all star stamps (bits 1|2|32: bright / crosses / faint squares).
    Rejects straps, edges, PS1 holes, TNS, asteroids, and any other set bit.
    """
    return (np.asarray(mask).astype(np.int64) & ~EPSF_IGNORE_BITS) != 0


def full_mask_bool(mask: np.ndarray) -> np.ndarray:
    """Boolean photutils / kernel / spatial mask: any set bit."""
    return np.asarray(mask) != 0


def hotpants_mask_bool(mask: np.ndarray) -> np.ndarray:
    """
    Boolean Hotpants mask: any set bit except ignored faint-catalog squares.

    Ignores bit 32 (``FAINT_CAT`` / Gaia mag ≳13 squares). Still masks bright
    catalog, sat crosses, straps, edges, PS1, TNS, asteroids, etc.
    """
    return (np.asarray(mask).astype(np.int64) & ~HOTPANTS_IGNORE_BITS) != 0


def strap_source_mask(mask: np.ndarray) -> np.ndarray:
    """Catalog sources for strap QE / phot_mask (bits 1|32)."""
    return (np.asarray(mask).astype(np.int64) & STRAP_SOURCE_BITS) != 0


def strap_column_mask(mask: np.ndarray) -> np.ndarray:
    """Strap columns (bit 4)."""
    return (np.asarray(mask).astype(np.int64) & STRAP) != 0
